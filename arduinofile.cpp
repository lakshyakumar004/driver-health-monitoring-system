#include <Wire.h>
#include "max86150.h"
MAX86150 max86150Sensor;
int16_t ecgsigned16;
int16_t redunsigned16;
uint16_t ppgunsigned16;
int valgsrsens;
int currtime;
void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  //Serial.println("MAX86150 Basic Reading Example");
  if (max86150Sensor.begin(Wire, I2C_SPEED_FAST) == false)
  {
    Serial.println("MAX86150 was not found. Please check wiring/power. ");
    while (1);
  }
  currtime = 0;
  max86150Sensor.setup();
}
void mainloop()
{
  valgsrsens=analogRead(A0);
  if (max86150Sensor.check() > 0)
  {
    ecgsigned16 = (int16_t)(max86150Sensor.getECG() >> 2);
    ppgunsigned16 = (uint16_t) (max86150Sensor.getFIFORed() >> 2);
    //Serial.print("GSR:");
    Serial.print(valgsrsens);
    Serial.print(" ");
    
//    Serial.print("ECG:");
//    Serial.print(ecgsigned16);
//    Serial.print(" ");
//    
    //Serial.print("PPG:");
    Serial.print(ppgunsigned16);
    Serial.println();
  }
  //delay(1000);
  
 }
void loop() {
  // put your main code here, to run repeatedly:
  
//  if(Serial.available()>0){
//    
//    String message = Serial.readStringUntil('\n');
//    message = message+" "+String(cnt);
//    cnt++;
//    Serial.println(message);
//    }
  if(millis()-currtime>=1000)
  {
      currtime = millis();
      mainloop();
  }
    //mainloop();
//    delay(500);
}
//#include <Wire.h>
//#include "max86150.h"
//MAX86150 max86150Sensor;
//uint16_t ppgunsigned16;
//
//void setup()
//{
//Serial.begin(57600);
//    Serial.println("MAX86150 PPG Streaming Example");
//
//    // Initialize sensor
//    if (max86150Sensor.begin(Wire, I2C_SPEED_FAST) == false)
//    {
//        Serial.println("MAX86150 was not found. Please check wiring/power. ");
//        while (1);
//    }
//
//    Serial.println(max86150Sensor.readPartID());
//
//    max86150Sensor.setup(); //Configure sensor. Use 6.4mA for LED drive
//}
//
//void loop()
//{
//    if(max86150Sensor.check()>0)
//    {
//        ppgunsigned16 = (uint16_t) (max86150Sensor.getFIFORed()>>2);
//        Serial.println(ppgunsigned16);
//        delay(100);
//    }
//    
//}