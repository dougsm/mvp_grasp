//ROS Dymo Scales Interface
//Based on HIDAPI example code

#include "ros/ros.h"
#include "std_msgs/Int16.h"
#include <hidapi/hidapi.h>
#include <sstream>


int main(int argc, char **argv)
{
  ros::init(argc, argv, "scales_interface");
  ros::NodeHandle n("~");
  ros::Publisher scales_pub = n.advertise<std_msgs::Int16>("weight", 1000);
  ros::Rate loop_rate(10);

  int res;
  unsigned char buf[65];
	#define MAX_STR 255
	wchar_t wstr[MAX_STR];
	hid_device *handle;

  res = hid_init();

  //Serial Number of device
  std::string serial_number;
  if (n.getParam("serial_number", serial_number)) {
      ROS_INFO_STREAM("Serial number received: " << serial_number);
  } else {
      ROS_WARN_STREAM("\"serial_number\" not provided.");
      return 0;
  }

  std::wstring serial( serial_number.begin(), serial_number.end() );
  const wchar_t* szSerial = serial.c_str();

  // Enumerate and print the HID devices on the system
  // struct hid_device_info *devs, *cur_dev;
  struct hid_device_info *cur_dev = hid_enumerate(0, 0);
  while (cur_dev) {
          printf("%04x %04x %s\n", cur_dev->vendor_id, cur_dev->product_id, cur_dev->serial_number);
          cur_dev = cur_dev->next;
  }

  // Open the scales using the VID, PID,
  // and optionally the Serial number. Replace with NULL for generic serial Number
  handle = hid_open(0x0922, 0x8003, szSerial);
  if(handle == NULL) {
    handle = hid_open(0x922, 0x8006, szSerial);
  }
  if(handle == NULL) {
    ROS_WARN_STREAM("COULDNT CONNECT TO SCALES\n");
    return 0;
  }

  // Read the Manufacturer String
  res = hid_get_manufacturer_string(handle, wstr, MAX_STR);
  printf("Manufacturer String: %ls\n", wstr);

  // Read the Product String
  res = hid_get_product_string(handle, wstr, MAX_STR);
  printf("Product String: %ls\n", wstr);

  // Read the Serial Number String
  res = hid_get_serial_number_string(handle, wstr, MAX_STR);
  printf("Serial Number String: %ls", wstr);
  printf("\n");

  int count = 0;
  while (ros::ok())
  {
    // printf("Hey, I'm here now\n");
    std_msgs::Int16 msg;
    msg.data = -1;

    // Read requested state
  	res = hid_read(handle, buf, 65);
    // printf("Made it past hid_read\n");
  	if (res < 0){
  		ROS_WARN("Unable to read() \n");
      //TODO implement trying to reconnect
      ros::shutdown(); //If read fails shutdown as there is no reconnect implemented
    }
    if(int(buf[2]) == 11) {
      msg.data = int((buf[4] | buf[5] << 8) * 2.83495);
      if(msg.data != 0) {
        ROS_WARN_STREAM("SCALES ARE IN POUNDS MODE, CONVERTING\n");
      }
    } else {
      msg.data = buf[4] | buf[5] << 8;
    }

    // ROS_INFO("Weight: %dg", msg.data);
    scales_pub.publish(msg);

    ros::spinOnce();

    loop_rate.sleep();
  }
  printf("I'm done now, good night\n");
  return 0;
}
