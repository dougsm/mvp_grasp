# Scales Interface

This package is for interfacing with DYMO digital scales over USB (e.g. [https://www.dymo.com/en-US/m10lb-digital-postal-scale](https://www.dymo.com/en-US/m10lb-digital-postal-scale)).  It is disabled by default so as to not introduce extra requirements for people who don't have the exact hardware.

To use this package:
1. remove CATKIN_IGNORE
2. install hidapi `sudo apt-get install libhidapi-dev`