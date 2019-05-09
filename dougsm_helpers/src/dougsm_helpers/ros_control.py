import rospy
import controller_manager_msgs.srv as cm_srv


class ControlSwitcher:
    """
    Class to simplify the act of switching between ROS controllers.
    Wraps an interface to /controller_manager/switch_controller.
    """
    def __init__(self, controllers, controller_manager_node='/controller_manager'):
        """
        :param controllers: Dictionary of controllers to manager/switch: {nick_name: controller_full_name}
        :param controller_manager_node: name of controller manager node.
        """
        self.controllers = controllers
        rospy.wait_for_service(controller_manager_node + '/switch_controller')
        rospy.wait_for_service(controller_manager_node + '/list_controllers')
        self.switcher_srv = rospy.ServiceProxy(controller_manager_node + '/switch_controller', cm_srv.SwitchController)
        self.lister_srv = rospy.ServiceProxy(controller_manager_node + '/list_controllers', cm_srv.ListControllers)

    def switch_controller(self, controller_name):
        """
        :param controller_name: Controller to activate.
        :return: Success True/False
        """
        rospy.sleep(0.5)
        start_controllers = [self.controllers[controller_name]]
        stop_controllers = [self.controllers[n] for n in self.controllers if n != controller_name]

        controller_switch_msg = cm_srv.SwitchControllerRequest()
        controller_switch_msg.strictness = 1
        controller_switch_msg.start_controllers = start_controllers
        controller_switch_msg.stop_controllers = stop_controllers

        res = self.switcher_srv(controller_switch_msg).ok
        if res:
            rospy.logdebug('Successfully switched to controller %s (%s)' % (controller_name, self.controllers[controller_name]))
            return res
        else:
            return False
