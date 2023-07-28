# Introduction
As it is essential to ensure food security and sustainable agriculture, plant disease detection is a crucial topic of research. The conventional approach to diagnosing plant diseases relies on trained individuals doing visual observations, which can be time-consuming, expensive, and prone to human error. The fusion of Internet of Things (IoT) and deep learning models has emerged as a viable strategy to automate the process of plant disease identification in order to address these difficulties.

The use of sensors and devices to gather information on environmental factors including temperature, humidity, and light intensity as well as plant health indicators like leaf colour, shape, and texture is referred to as IoT-based plant disease detection.

After being transmitted to a central processing unit, this data is analysed and categorised into several plant disease categories using deep learning models. Convolutional neural networks (CNNs), one type of deep learning model, have demonstrated significant promise for effectively identifying and categorising plant diseases. The accuracy and speed of disease diagnosis are improved by these models' ability to extract intricate patterns and features from huge datasets. The deployment of IoT devices also makes it possible to monitor plant health in real-time, which enables early identification and action, lowering the risk of crop loss and increasing yields.

For farmers who manage huge tracts of land, the adoption of IoT devices also enables remote crop monitoring. Large databases of photos of healthy and diseased plants can be used to train deep learning models, which can increase the precision of disease classification. The construction of prediction models that can foretell the possibility of disease outbreaks based on environmental data can also be facilitated by the merging of IoT and deep learning algorithms. IoT devices can be used to optimize irrigation and fertilizer use in addition to detecting plant illnesses, which can increase crop yields and lessen environmental impact. The expense of IoT devices and the requirement for dependable and quick internet connectivity in rural regions remain obstacles despite the potential advantages of this technology.
In conclusion, the fusion of IoT and deep learning models offers a viable strategy to transform the detection of plant diseases. With the right application, this technology might increase the speed and accuracy of illness diagnosis, lower labour costs, and support sustainable agriculture.

# Problem Statement
Despite the efforts of farmers and agriculturists, plant diseases still cause significant losses in crop yields every year. These losses can be caused by a variety of factors, including climate change, lack of proper care, and new or emerging diseases. Early detection and diagnosis of plant diseases is crucial for reducing these losses, but current methods can be timeconsuming and costly.
To develop a system for early detection and diagnosis of plant diseases using IoT technology. To collect and analyze data on plant growth, health, and symptoms to identify any signs of disease or stress. To create an automated and efficient way to detect and diagnose plant diseases in order to improve crop yields and reduce losses due to diseases. To study the effect of climate change on plant diseases, and to detect new or emerging diseases. To provide farmers with real-time monitoring capabilities of their crop and allow them to take early action to prevent losses.

# Hardware/Software Description
1.	Arduino: A microcontroller board called Arduino Uno is used. It contains a 16 MHz quartz crystal, 6 analogue inputs, 14 digital input/output pins (of which 6 may be used as PWM outputs), a USB port, a power connector, an ICSP header, and a reset button. It comes with everything needed to support the microcontroller; to get started, just plug in a USB cable, an AC-to-DC converter, or a battery. 
![image](https://github.com/Shyam301910/Plant-Disease-Detection-using-IoT/assets/95332840/e81b1f13-eb16-4c38-9661-95fffbe6b748)

2.	Python : It monitors the serial communication out from the Arduino and obtain the data sent by the Arduino and send to cloud service as HTTP Request. We perform this using TCP/IP protocol communication via HTTP Request.
   
3.	ESP32 Cam Module: ESP32 is a line of inexpensive, low-power system on a chip microcontroller that include built-in dual-mode Bluetooth and Wi-Fi. The board includes two powerful 32-bit LX6 Processors, WiFi, classic Bluetooth, and low power BLE. The main frequency adjustment runs from 80MHz to 240MHz, and it uses a 7-stage pipeline architecture, an on-chip sensor, a Hall sensor, a temperature sensor, and other sensors. The ESP32-CAM is widely applicable in numerous IoT applications. It is appropriate for IoT applications such as wireless positioning system signals, industrial wireless control, wireless monitoring, QR wireless identification, and smart home gadgets. For IoT applications, it is the perfect answer.
![image](https://github.com/Shyam301910/Plant-Disease-Detection-using-IoT/assets/95332840/70192e2d-c5c6-4343-879b-5341a398f162)

4.	Bread Board: An electronic circuit can be built and tested using a breadboard, commonly referred to as a prototyping board or solderless breadboard. It typically consists of a grid of holes arranged in rows and columns and is made of plastic. These holes are used to insert jumper wires to link electronic components like resistors, capacitors, and integrated circuits. Below the holes on the breadboard are metal clips or sockets that make it simple to insert and remove components. Moreover, the sockets or clips offer a way to join parts collectively without the need of solder. As a result, engineers, students, and hobbyists who want to quickly develop or test electronic circuits frequently choose breadboards. 
Typically, the breadboard is divided into two parts: the top and the bottom. Components are located in the top section, while power and ground rails are found in the bottom section. The components installed onto the breadboard are powered by these rails, which run the length of the breadboard. The positive and negative rails are commonly denoted by the colors red and black, respectively, on the rails. Breadboards are available in a variety of shapes and sizes, from tiny little breadboards to sizable breadboards that can support intricate circuits. To test and debug electronic circuits, they are frequently used in conjunction with other equipment like oscilloscopes and multimeters.
![image](https://github.com/Shyam301910/Plant-Disease-Detection-using-IoT/assets/95332840/455f0f1e-b114-42a2-9968-2b0e312132e5)

5.	Jumper cables:  Jumper cables are a form of electrical wire that are used to establish transient connections between elements or points on a circuit board or breadboard. They are sometimes referred to as jumper wires or wire jumpers. Typically, they are constructed from flexible stranded wire with connectors at each end, allowing them to be quickly inserted into openings on a breadboard or in other parts. Jumper cables are available in a range of diameters, colors, and lengths to suit various purposes. They can be used to provide connections between various locations on a circuit board or to join components like resistors, capacitors, and integrated circuits. Depending on the application and the type of component being connected, the connections on the ends of the jumper cables may be pins, sockets, or alligator clips. Jumper cables are frequently used in electronics prototype and testing because they make it simple and quick to connect components without soldering. They can be used to test specific circuit components or to bypass malfunctioning ones, which makes them useful for troubleshooting and debugging. 
Jumper cables should not be used for permanent wiring as they are designed for temporary connections only. Use of proper wiring methods and materials that adhere to safety regulations and electrical codes is crucial for permanent wiring.
![image](https://github.com/Shyam301910/Plant-Disease-Detection-using-IoT/assets/95332840/a519ef2b-e5ce-4129-b5d8-3d992c326fff)

# Architecture 
![image](https://github.com/Shyam301910/Plant-Disease-Detection-using-IoT/assets/95332840/13032c82-ae4c-4936-a271-a6893e9e4d03)

# Implementation
The suggested system is a working prototype for Internet of Things-based plant disease detection. The hardware consists of an esp32 cam module and an Arduino Uno. The esp32 cam module receives power supply from the Arduino. On the software side, we combined two well-known models to construct an ML model utilising ensemble learning that can determine if a plant has illness or not given a picture of a leaf as input. We've designed a straightforward frontend page for users to interact with the system in addition to hosting this machine learning model on a flask server. The server that houses the ML model is first started. After this, we connect the arduino to the pc which starts up the esp32 cam module. We link our system to the local access point that the esp32 establishes. The frontend component is therefore how we communicate with the system. The so-called "Capture & Predict" button causes the esp32 cam module to take a picture, which is then uploaded to the ML model for prediction by the server. The ML model then generates a result based on the input image and sends it back to the server. For the user's benefit, the server then shows the forecast together with its accuracy % on the front end.

![image](https://github.com/Shyam301910/Plant-Disease-Detection-using-IoT/assets/95332840/ef02d9d4-cf57-48be-9fc5-396b2c87da58)

# Model acccuracy comparison
![image](https://github.com/Shyam301910/Plant-Disease-Detection-using-IoT/assets/95332840/243e0b29-942b-4c1a-8598-53ac6e8f8e71)

# Circuit and Model outcomes images
![image](https://github.com/Shyam301910/Plant-Disease-Detection-using-IoT/assets/95332840/880e9f6d-0ad1-411a-9e19-2322cf9a1e33)

![image](https://github.com/Shyam301910/Plant-Disease-Detection-using-IoT/assets/95332840/866977b9-b96a-4677-934b-405bf6ce8ff1)

![image](https://github.com/Shyam301910/Plant-Disease-Detection-using-IoT/assets/95332840/fef7eec9-dac6-42d5-abd7-3e2399b61bc2)

![image](https://github.com/Shyam301910/Plant-Disease-Detection-using-IoT/assets/95332840/1c1a2c0c-3e2d-4864-afe6-f037b0b41a8b)

# Video Demontration
Video Demonstration Link - https://drive.google.com/file/d/19Juns4oFmMt_pNo5OzrzToX8xk1l50rg/view?usp=share_link
