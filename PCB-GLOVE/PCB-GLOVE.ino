//#define NO_OTA_PORT
#include<Wire.h>   
#include <ESP8266WiFi.h>
#include <ArduinoOTA.h>
#include <stdlib.h>
#include <FS.h>
#define LED D0

//ESP8266WebServer server(80);    // Create a webserver object that listens for HTTP request on port 80

File fsUploadFile; 

//const char* ssid     = "SpectrumSetup-9A";        // The SSID (name) of the Wi-Fi network you want to connect to
//const char* password = "everywhile736";     // The password of the Wi-Fi network
 
int S0A=D3;   // defining each pin of MCU used 
int S1A=D4;
int S2A=D5;
int S0B=D6;
int S1B=D7;
int S2B=D8;
int i;
int j=1;
int LDC = 0x2A;  // when the extra part of LDC is removed 

//int CH0MSB = 0x00;
//
//int CH0LSB = 0x01;

int CH1MSB = 0x02;

int CH1LSB = 0x03;

//int CH2MSB = 0x04;
//
//int CH2LSB = 0x05;
//
//int CH3MSB = 0x06;
//
//int CH3LSB = 0x07;
//
//long initial2 = 0;
//
  long initial1 = 0;
//
//long initial0 = 0;
//
//long initial3 = 0;


//unsigned long readChannel0()
//
//{
//
//  unsigned long val = 0;
//
//  word c = 0; //a word stores a 16-bit unsigned number
//
//  word d = 0;
//
//  c = readValue(LDC, CH0MSB);
//
//  d = readValue(LDC, CH0LSB);
//
//  val = c;
//
//  val <<= 16;
//
//  val += d;
//
//  return val;
//
//}

 

 

//unsigned long readChannel3()
//
//{
//
//  unsigned long val = 0;
//
//  word c = 0; //a word stores a 16-bit unsigned number
//
//  word d = 0;
//
//  c = readValue(LDC, CH3MSB);
//
//  d = readValue(LDC, CH3LSB);
//
//  val = c;
//
//  val <<= 16;
//
//  val += d;
//
//  return val;
//
//}

 

 

unsigned long readChannel1()

{

  unsigned long val = 0;

  word c = 0; //a word stores a 16-bit unsigned number

  word d = 0;

  c = readValue(LDC, CH1MSB);

  d = readValue(LDC, CH1LSB);

  val = c;

  val <<= 16;

  val += d;

  return val;

}

 

//unsigned long readChannel2()
//
//{
//
//  unsigned long val = 0;
//
//  word c = 0;
//
//  word d = 0;
//
//  c = readValue(LDC, CH2MSB);
//
//  d = readValue(LDC, CH2LSB);
//
//  val = c;
//
//  val <<= 16;
//
//  val += d;
//
//  return val;
//
//}

 

void Calibrate()

{

//  initial0 = readChannel0();

  initial1 = readChannel1();

//  initial2 = readChannel2();
//
//  initial3 = readChannel3();

}

 

word readValue (int LDC, int reg)

{

  int a = 0;

  int b = 0;

  word value = 0;

  Wire.beginTransmission(LDC);


  Wire.write(reg);

  Wire.endTransmission();

  Wire.requestFrom(LDC, 2);

  while (Wire.available())

  {

    a = Wire.read();

    b = Wire.read();

  }

  value = a;

  value <<= 8;

  value += b;

  return value;

}

 

 

void writeConfig(int LDC, int reg, int MSB, int LSB)

{

  Wire.beginTransmission(LDC);

  Wire.write(reg);

  Wire.write(MSB);

  Wire.write(LSB);

  Wire.endTransmission();

}

 

void Configuration()

{

//  writeConfig(LDC, 0x14, 0x10, 0x02);//CLOCK_DIVIDERS_CH0
//
//  writeConfig(LDC, 0x1E, 0x90, 0x00);//DRIVE_CURRENT_CH0
//
//  writeConfig(LDC, 0x10, 0x00, 0x0A);//SETTLECOUNT_CH0
//
//  writeConfig(LDC, 0x08, 0x04, 0xD6);//RCOUNT_CH0

  writeConfig(LDC, 0x15, 0x10, 0x02);//CLOCK_DIVIDERS_CH1

  writeConfig(LDC, 0x1F, 0x90, 0x00);//DRIVE_CURRENT_CH1

  writeConfig(LDC, 0x11, 0x00, 0x0A);//SETTLECOUNT_CH1

  writeConfig(LDC, 0x09, 0x04, 0xD6);//RCOUNT_CH1

//  writeConfig(LDC, 0x16, 0x10, 0x02);//CLOCK_DIVIDERS_CH2
//
//  writeConfig(LDC, 0x20, 0x90, 0x00);//DRIVE_CURRENT_CH2
//
//  writeConfig(LDC, 0x12, 0x00, 0x0A);//SETTLECOUNT_CH2
//
//  writeConfig(LDC, 0x0A, 0x04, 0xD6);//RCOUNT_CH2
//
//  writeConfig(LDC, 0x17, 0x10, 0x02);//CLOCK_DIVIDERS_CH3
//
//  writeConfig(LDC, 0x21, 0x90, 0x00);//DRIVE_CURRENT_CH3
//
//  writeConfig(LDC, 0x13, 0x00, 0x0A);//SETTLECOUNT_CH3
//
//  writeConfig(LDC, 0x0B, 0x04, 0xD6);//RCOUNT_CH3

  writeConfig(LDC, 0x19, 0x00, 0x00);//ERROR_CONFIG

  writeConfig(LDC, 0x1B, 0xC2, 0x0C);//MUX_CONFIG

  writeConfig(LDC,0x1A,0x08,0x01); // SLEEP_MODE_EN  // when extra part of LDC is removed 

}

const int numsamples = 50;
unsigned long a1[numsamples];
unsigned long a2[numsamples];
unsigned long a3[numsamples];
unsigned long a4[numsamples];
unsigned long a5[numsamples];
unsigned long a6[numsamples];
unsigned long a7[numsamples];
unsigned long a8[numsamples];
int sampleindex1 = 0;
int sampleindex2 = 0;
int sampleindex3 = 0;
int sampleindex4 = 0;
int sampleindex5 = 0;
int sampleindex6 = 0;
int sampleindex7 = 0;
int sampleindex8 = 0;
void setup()

{
 Serial.begin(9600);
// while (!Serial)
// {
// }

// sets pins as OUTPUT 
pinMode(S0A, OUTPUT);  
pinMode(S1A, OUTPUT); 
pinMode(S2A, OUTPUT);   
pinMode(S0B, OUTPUT); 
pinMode(S1B, OUTPUT);   
pinMode(S2B, OUTPUT);   
pinMode(LED, OUTPUT);
// Serial.begin(9600);         // Start the Serial communication to send messages to the computer
//delay(20);
  //Serial.println('\n');
  
//  Wire.begin(ssid, password);             // Connect to the network
 // Serial.print("Connecting to ");
  //Serial.print(ssid); 
  //Serial.println(" ...");

 i=0;
  Wire.begin();

// Serial.begin(9600);
  Configuration();
  

  delay(500);

  Calibrate();
 
  
}


 

/**void loop()

{

  unsigned long data0 = readChannel0();

  unsigned long data1 = readChannel1();

  unsigned long data2 = readChannel2();

  unsigned long data3 = readChannel3();

  Serial.println(data0);

  Serial.println(data1);

  Serial.println(data2);

  Serial.println(data3);

  delay(20);

}**/
void out1()
{
  // put the pins of multipexer HIGH/LOW based on the connection
  // A0 of Multiplexer A and A0 of Multiplexer B (first coil,Thumb)
  
  digitalWrite(S0A,LOW);
  digitalWrite(S1A,LOW);
  digitalWrite(S2A,LOW);
  digitalWrite(S0B,LOW);  
  digitalWrite(S1B,LOW);
  digitalWrite(S2B,LOW);
}

void out2()
{
   // A1 of Multiplexer A and A1 of Multiplexer B (second coil,Index)
  digitalWrite(S0A,HIGH);
  digitalWrite(S1A,LOW);
  digitalWrite(S2A,LOW);
  digitalWrite(S0B,HIGH);  
  digitalWrite(S1B,LOW);
  digitalWrite(S2B,LOW); 
}

void out3()  // when  
{
  // A2 of Multiplexer A and A2 of Multiplexer B (third coil,Middle)
  digitalWrite(S0A,LOW);
  digitalWrite(S1A,HIGH);
  digitalWrite(S2A,LOW);
  digitalWrite(S0B,LOW);  
  digitalWrite(S1B,HIGH);
  digitalWrite(S2B,LOW); 
}

void out4()  // when  
{
  // A3 of Multiplexer A and A3 of Multiplexer B (second coil,Ring)
  digitalWrite(S0A,HIGH);
  digitalWrite(S1A,HIGH);
  digitalWrite(S2A,LOW);
  digitalWrite(S0B,HIGH);  
  digitalWrite(S1B,HIGH);
  digitalWrite(S2B,LOW); 
}

void out5()  // when 
{
  // A4 of Multiplexer A and A4 of Multiplexer B (second coil,Pinky)
  digitalWrite(S0A,LOW);
  digitalWrite(S1A,LOW);
  digitalWrite(S2A,HIGH);
  digitalWrite(S0B,LOW);  
  digitalWrite(S1B,LOW);
  digitalWrite(S2B,HIGH); 
}

void out6()  // when 
{
  // A5 of Multiplexer A and A5 of Multiplexer B (second coil,palm)
  digitalWrite(S0A,HIGH);
  digitalWrite(S1A,LOW);
  digitalWrite(S2A,HIGH);
  digitalWrite(S0B,HIGH);  
  digitalWrite(S1B,LOW);
  digitalWrite(S2B,HIGH); 
}

void out7()  // when 
{
  // A6 of Multiplexer A and A6 of Multiplexer B (second coil,wrist)
  digitalWrite(S0A,LOW);
  digitalWrite(S1A,HIGH);
  digitalWrite(S2A,HIGH);
  digitalWrite(S0B,LOW);  
  digitalWrite(S1B,HIGH);
  digitalWrite(S2B,HIGH); 
}

void out8()  // when 
{
  // A7 of Multiplexer A and A7 of Multiplexer B (second coil,ulnar)
  digitalWrite(S0A,HIGH);
  digitalWrite(S1A,HIGH);
  digitalWrite(S2A,HIGH);
  digitalWrite(S0B,HIGH);  
  digitalWrite(S1B,HIGH);
  digitalWrite(S2B,HIGH); 
}
/*
 // this codes applies when we directly read from all the channels of LDC1614
void loop()

{
 unsigned long data0 = readChannel0();
 Serial.println((String)"1"); 
 Serial.println( + data0);
 delay(20);
 unsigned long data1 = readChannel1();
 Serial.println((String)"2"); 
  Serial.println( + data1);
  delay(20);
 unsigned long data2 = readChannel2();
Serial.println((String)"3"); 
  Serial.println( + data2);
  delay(20); 
 unsigned long data3 = readChannel3();
 Serial.println((String)"4"); 
  Serial.println( + data3);
  delay(20);
 }*/

//void beep(){
//  tone(9,1000,100);
//}
 void loop()
{

if (Serial.available()>0)
{
 while (sampleindex8 < 50 ){
 for(i = 0; i < 8; i++)
 {
 if(i==0)
 {
 out3();  // Thumb read the coil connected to the multiplexer  when first two pins are HIGH; two coils of the multiplexer are read through channel 3,
 delay(20);
 unsigned long data1 = readChannel1();
 Serial.println((String)"1");  // prints that string "1" that we read it later in MATLAB to know from which sensor we are reading; 1- corresponds to sensor 1
 Serial.println(+ data1);
 a1[sampleindex1++] = data1;
 sampleindex1= sampleindex1++;
// Serial.write(data1);
 delay(20);
 }
 else if (i==1)
 {
  out2();   // Index multiplexer pins all LOW; Reads directly from LDC channel 0 
  delay(20);
  unsigned long data2 = readChannel1();
  Serial.println((String)"2" );  // prints that string "2" that we read it later in MATLAB to know from which sensor we are reading; 2- corresponds to sensor 2
  Serial.println(+ data2);
  a2[sampleindex2++] = data2;
  sampleindex2= sampleindex2++;
 // Serial.write(data2);
  delay(20);
 }
 else if (i==2)
 {
  out1();  // Middle multiplexer pins all LOW; Reads directly from LDC channel 1
  delay(20);
  unsigned long data3 = readChannel1();
  Serial.println((String)"3" );
  Serial.println(+ data3); 
   a3[sampleindex3++] = data3; 
   sampleindex3= sampleindex3++;
//   Serial.write(data3);
  delay(20);
  }
  else if (i==3)
  {
  out4();// Ring  multiplexer pins all LOW; Reads directly from LDC channel 1
  delay(20);
  unsigned long data4 = readChannel1();
  Serial.println((String)"4" );
  Serial.println(+ data4);
  a4[sampleindex4++] = data4;
  sampleindex4= sampleindex4++;
//  Serial.write(data4);
  delay(20);
  }
 else if (i==4)
 {
  out5();  // Pinky switchig to other coil connected through multuplexer; first two pins LOW, then the rest HIGH 
  delay(20);
  unsigned long data5 = readChannel1();
  Serial.println((String)"5" );   // prints that string "5" that we read it later in MATLAB to know from which sensor we are reading; 5- corresponds to coil 5
  Serial.println(+ data5);
  a5[sampleindex5++] = data5;
  sampleindex5= sampleindex5++;
//   Serial.write(data5);
  delay(20);
 }
 else if (i==5)
 {
  out7();   // Palm multiplexer pins all LOW; Reads directly from LDC channel 0 
  delay(20);
  unsigned long data6 = readChannel1();
  Serial.println((String)"6" );  // prints that string "2" that we read it later in MATLAB to know from which sensor we are reading; 2- corresponds to sensor 2
  Serial.println(+ data6);
   a6[sampleindex6++] = data6;
   sampleindex6= sampleindex6++;
//   Serial.write(data6);
  delay(20);
 }
 else if (i==6)
 {
  out8();   // Wrist multiplexer pins all LOW; Reads directly from LDC channel 0 
  delay(20);
  unsigned long data7 = readChannel1();
  Serial.println((String)"7" );  // prints that string "2" that we read it later in MATLAB to know from which sensor we are reading; 2- corresponds to sensor 2
  Serial.println(+ data7);
  a7[sampleindex7++] = data7;
  sampleindex7= sampleindex7++;
//  Serial.write(data7);
  delay(20);
 }
 else if (i==7)
 {
  out6();   // Ulnar multiplexer pins all LOW; Reads directly from LDC channel 0 
  delay(20);
  unsigned long data8 = readChannel1();
  Serial.println((String)"8" );  // prints that string "2" that we read it later in MATLAB to know from which sensor we are reading; 2- corresponds to sensor 2
  Serial.println(+ data8);
  a8[sampleindex8++] = data8;
  sampleindex8= sampleindex8++;
//  Serial.write(data8);
  
  delay(20);
 }
 if (sampleindex1 % 5 ==0 && sampleindex2 % 5 ==0 &&  sampleindex3 % 5 ==0 && sampleindex4 % 5 ==0 && sampleindex5 % 5 ==0 && sampleindex6 % 5 ==0 && sampleindex7 % 5 ==0 && sampleindex8 % 5 ==0)
 {
 //Serial.println(+ "beeppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp");
  digitalWrite(LED, LOW);
  delay(3000);
  digitalWrite(LED, HIGH);
 }
 
  
// if (sampleindex1 <= 100 && sampleindex2 <=100 &&  sampleindex3 <= 100  && sampleindex4 <=100 && sampleindex5 <=100 && sampleindex6 <=100 && sampleindex7 <=100 && sampleindex8 <=100){
//  
//  break;
   }
  }
  sampleindex1 = 0;
  sampleindex2 = 0;
  sampleindex3 = 0;
  sampleindex4 = 0;
  sampleindex5 = 0;
  sampleindex6 = 0;
  sampleindex7 = 0;
  sampleindex8 = 0;
} 
}
