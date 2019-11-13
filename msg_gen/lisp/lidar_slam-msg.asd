
(cl:in-package :asdf)

(defsystem "lidar_slam-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :std_msgs-msg
)
  :components ((:file "_package")
    (:file "CobotOdometryMsg" :depends-on ("_package_CobotOdometryMsg"))
    (:file "_package_CobotOdometryMsg" :depends-on ("_package"))
  ))