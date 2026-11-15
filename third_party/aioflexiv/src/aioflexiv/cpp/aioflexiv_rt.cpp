#include <flexiv/rdk/data.hpp>
#include <flexiv/rdk/model.hpp>
#include <flexiv/rdk/mode.hpp>
#include <flexiv/rdk/robot.hpp>

#include <Eigen/Eigen>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
namespace rdk = flexiv::rdk;

namespace {

bool TimestampChanged(
    const std::pair<int, int>& lhs, const std::pair<int, int>& rhs)
{
    return lhs.first != rhs.first || lhs.second != rhs.second;
}

void LogStartup(const std::string& message)
{
    std::cout << "[aioflexiv] " << message << std::endl;
}

void LogStartupError(const std::string& message)
{
    std::cerr << "[aioflexiv] " << message << std::endl;
}

void EnsureOperational(rdk::Robot& robot, int timeout_sec = 30, bool auto_clear_fault = true,
    unsigned int fault_clear_timeout_sec = 30)
{
    if (robot.operational()) {
        return;
    }

    if (robot.recovery()) {
        LogStartupError(
            "robot is in recovery state; refusing to run automatic recovery during start()");
        throw std::runtime_error(
            "Flexiv robot is in recovery state. Automatic recovery can move joints and "
            "requires a reboot afterward; run Robot::RunAutoRecovery() intentionally instead.");
    }

    bool cleared_fault = false;
    if (robot.fault()) {
        LogStartup("detected robot fault before startup");
        if (!auto_clear_fault) {
            LogStartupError("auto_clear_fault is disabled; leaving robot in fault state");
            throw std::runtime_error(
                "Flexiv robot is in fault state and auto_clear_fault is disabled");
        }
        LogStartup("attempting ClearFault(timeout_sec="
            + std::to_string(fault_clear_timeout_sec) + ")");
        const bool clear_fault_ok = robot.ClearFault(fault_clear_timeout_sec);
        const bool still_in_fault = robot.fault();
        if (!clear_fault_ok || still_in_fault) {
            LogStartupError("ClearFault failed: return="
                + std::string(clear_fault_ok ? "true" : "false")
                + ", fault_after="
                + std::string(still_in_fault ? "true" : "false"));
            throw std::runtime_error("Failed to clear Flexiv robot fault");
        }
        cleared_fault = true;
        LogStartup("fault cleared successfully");
    }

    if (cleared_fault) {
        LogStartup("enabling robot after fault clear");
    }
    robot.Enable();

    if (cleared_fault) {
        LogStartup("waiting for robot to become operational (timeout_sec="
            + std::to_string(timeout_sec) + ")");
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    while (std::chrono::steady_clock::now() < deadline) {
        if (robot.recovery()) {
            LogStartupError(
                "robot entered recovery state while waiting to become operational");
            throw std::runtime_error(
                "Flexiv robot entered recovery state while waiting to become operational");
        }
        if (robot.fault()) {
            LogStartupError(
                "robot entered fault state while waiting to become operational");
            throw std::runtime_error(
                "Flexiv robot entered fault state while waiting to become operational");
        }
        if (robot.operational()) {
            if (cleared_fault) {
                LogStartup("robot is operational");
            }
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (cleared_fault) {
        LogStartupError("timed out waiting for robot to become operational");
    }
    throw std::runtime_error("Timed out waiting for Flexiv robot to become operational");
}

class ActiveTorqueControl {
public:
    ActiveTorqueControl(const std::string& robot_sn,
        const std::vector<std::string>& network_interface_whitelist = {}, bool verbose = true,
        bool auto_clear_fault = true, unsigned int fault_clear_timeout_sec = 30,
        int operational_timeout_sec = 30)
        : robot_(robot_sn, network_interface_whitelist, verbose)
        , model_(robot_)
    {
        EnsureOperational(
            robot_, operational_timeout_sec, auto_clear_fault, fault_clear_timeout_sec);
        robot_.SwitchMode(rdk::Mode::RT_JOINT_TORQUE);
        last_timestamp_ = robot_.states().timestamp;
    }

    rdk::RobotStates read_once(int timeout_ms = 1000)
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

        while (std::chrono::steady_clock::now() < deadline) {
            auto state = robot_.states();
            if (TimestampChanged(state.timestamp, last_timestamp_)) {
                last_timestamp_ = state.timestamp;
                return state;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }

        throw std::runtime_error("Timed out waiting for a fresh Flexiv robot state");
    }

    void write_once(const std::vector<double>& torques, bool enable_gravity_comp = true,
        bool enable_soft_limits = true, double friction_comp_scale = 100.0)
    {
        robot_.StreamJointTorque(
            torques, enable_gravity_comp, enable_soft_limits, friction_comp_scale);
    }

    rdk::RobotStates states() const { return robot_.states(); }
    void stop() { robot_.Stop(); }

    py::dict info() const
    {
        const auto info = robot_.info();

        py::dict result;
        result["serial_num"] = info.serial_num;
        result["software_ver"] = info.software_ver;
        result["model_name"] = info.model_name;
        result["license_type"] = info.license_type;
        result["dof"] = info.DoF;
        result["manipulator_dof"] = info.DoF_m;
        result["q_min"] = info.q_min;
        result["q_max"] = info.q_max;
        result["dq_max"] = info.dq_max;
        result["tau_max"] = info.tau_max;
        result["K_q_nom"] = info.K_q_nom;
        result["K_x_nom"] = info.K_x_nom;
        result["has_FT_sensor"] = info.has_FT_sensor;
        return result;
    }

    py::dict read_once_full(const std::string& link_name = "flange", int timeout_ms = 1000)
    {
        const auto state = read_once(timeout_ms);
        model_.Update(state.q, state.dtheta);

        py::dict result;
        result["timestamp"] = state.timestamp;
        result["qpos"] = state.q;
        result["qvel"] = state.dtheta;
        result["dq"] = state.dq;
        result["theta"] = state.theta;
        result["dtheta"] = state.dtheta;
        result["tau"] = state.tau;
        result["tau_des"] = state.tau_des;
        result["tau_ext"] = state.tau_ext;
        result["tcp_pose"] = state.tcp_pose;
        result["tcp_vel"] = state.tcp_vel;
        result["flange_pose"] = state.flange_pose;
        result["ft_sensor_raw"] = state.ft_sensor_raw;
        result["ext_wrench_in_tcp"] = state.ext_wrench_in_tcp;
        result["ext_wrench_in_world"] = state.ext_wrench_in_world;
        result["ext_wrench_in_tcp_raw"] = state.ext_wrench_in_tcp_raw;
        result["ext_wrench_in_world_raw"] = state.ext_wrench_in_world_raw;
        result["ee"] = model_.T(link_name).matrix();
        result["jac"] = model_.J(link_name);
        result["mm"] = model_.M();
        result["coriolis"] = model_.c();
        result["gravity"] = model_.g();
        return result;
    }

private:
    rdk::Robot robot_;
    rdk::Model model_;
    std::pair<int, int> last_timestamp_ {};
};

py::dict compile_time_rt_probe()
{
    auto switch_mode = &rdk::Robot::SwitchMode;
    auto states = &rdk::Robot::states;
    auto stream_torque = &rdk::Robot::StreamJointTorque;
    auto fault = &rdk::Robot::fault;
    auto recovery = &rdk::Robot::recovery;
    auto clear_fault = &rdk::Robot::ClearFault;
    auto model_mass = &rdk::Model::M;
    auto model_jacobian = &rdk::Model::J;

    py::dict result;
    result["rt_joint_torque_mode_value"] = static_cast<int>(rdk::Mode::RT_JOINT_TORQUE);
    result["has_switch_mode_symbol"] = switch_mode != nullptr;
    result["has_states_symbol"] = states != nullptr;
    result["has_stream_joint_torque_symbol"] = stream_torque != nullptr;
    result["has_robot_fault_symbol"] = fault != nullptr;
    result["has_robot_recovery_symbol"] = recovery != nullptr;
    result["has_clear_fault_symbol"] = clear_fault != nullptr;
    result["has_model_mass_symbol"] = model_mass != nullptr;
    result["has_model_jacobian_symbol"] = model_jacobian != nullptr;
    return result;
}

} // namespace

PYBIND11_MODULE(_aioflexiv_rt, m)
{
    m.doc() = "Tiny Flexiv RT torque-control bridge";

    py::class_<rdk::RobotStates>(m, "RobotStates")
        .def_readonly("timestamp", &rdk::RobotStates::timestamp)
        .def_readonly("q", &rdk::RobotStates::q)
        .def_readonly("theta", &rdk::RobotStates::theta)
        .def_readonly("dq", &rdk::RobotStates::dq)
        .def_readonly("dtheta", &rdk::RobotStates::dtheta)
        .def_readonly("tau", &rdk::RobotStates::tau)
        .def_readonly("tau_des", &rdk::RobotStates::tau_des)
        .def_readonly("tau_ext", &rdk::RobotStates::tau_ext)
        .def_readonly("tcp_pose", &rdk::RobotStates::tcp_pose)
        .def_readonly("tcp_vel", &rdk::RobotStates::tcp_vel)
        .def_readonly("flange_pose", &rdk::RobotStates::flange_pose)
        .def_readonly("ft_sensor_raw", &rdk::RobotStates::ft_sensor_raw)
        .def_readonly("ext_wrench_in_tcp", &rdk::RobotStates::ext_wrench_in_tcp)
        .def_readonly("ext_wrench_in_world", &rdk::RobotStates::ext_wrench_in_world)
        .def_readonly("ext_wrench_in_tcp_raw", &rdk::RobotStates::ext_wrench_in_tcp_raw)
        .def_readonly("ext_wrench_in_world_raw", &rdk::RobotStates::ext_wrench_in_world_raw);

    py::class_<ActiveTorqueControl>(m, "ActiveTorqueControl")
        .def(py::init<const std::string&, const std::vector<std::string>&, bool, bool,
                 unsigned int, int>(),
            py::arg("robot_sn"), py::arg("network_interface_whitelist") = std::vector<std::string> {},
            py::arg("verbose") = true, py::arg("auto_clear_fault") = true,
            py::arg("fault_clear_timeout_sec") = 30, py::arg("operational_timeout_sec") = 30)
        .def("read_once", &ActiveTorqueControl::read_once, py::arg("timeout_ms") = 1000,
            py::call_guard<py::gil_scoped_release>())
        .def("write_once", &ActiveTorqueControl::write_once, py::arg("torques"),
            py::arg("enable_gravity_comp") = true, py::arg("enable_soft_limits") = true,
            py::arg("friction_comp_scale") = 100.0, py::call_guard<py::gil_scoped_release>())
        .def("states", &ActiveTorqueControl::states)
        .def("info", &ActiveTorqueControl::info)
        .def("read_once_full", &ActiveTorqueControl::read_once_full, py::arg("link_name") = "flange",
            py::arg("timeout_ms") = 1000)
        .def("stop", &ActiveTorqueControl::stop);

    m.def("compile_time_rt_probe", &compile_time_rt_probe);
}
