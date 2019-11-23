
(cl:in-package :asdf)

(defsystem "lidar_slam-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
               :std_msgs-msg
)
  :components ((:file "_package")
    (:file "CobotOdometryMsg" :depends-on ("_package_CobotOdometryMsg"))
    (:file "_package_CobotOdometryMsg" :depends-on ("_package"))
    (:file "HitlSlamInputMsg" :depends-on ("_package_HitlSlamInputMsg"))
    (:file "_package_HitlSlamInputMsg" :depends-on ("_package"))
  ))