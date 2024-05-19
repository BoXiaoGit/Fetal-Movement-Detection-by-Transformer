# Fetal-Movement-Detection-by-Transformer
This project uses a transformer-based AI model to detect fetal movements in the womb.
This tool processes data related to fetal movements, utilizing sensor readings and corresponding annotations from ultrasound observations to prepare a structured dataset. The primary goal is to correlate sensor data with expert annotations to facilitate detailed analysis of fetal behavior and movement patterns.

Data Structure and Collection:

Sensor Data (CSV File): The raw data is collected via various sensors attached to a mother's abdomen. These sensors capture a wide range of physiological signals including mechanical responses of muscles and movement intensity. The data includes:

Sequential Counts: An index for each data point.
Sensor Readings: Numerical values from sensors like piezoelectric sensors and mechanomyography sensors, which provide data on muscle and movement dynamics.
Accelerometer Data: X, Y, and Z axis readings capturing motion and orientation.
Inertial Measurement Units (IMU): Data from accelerometers, gyroscopes, and magnetometers that give comprehensive insights into the orientation, acceleration, and magnetic fields around the fetus.
Annotations Data (Excel File): Annotations are provided by medical professionals based on ultrasound imaging. This data complements the sensor readings by marking specific types of fetal movements with precise timestamps, facilitating a deeper understanding of the sensor data. The annotations include:

Timestamps: Both relative (in seconds from the start of the session) and specific (date and time).
Movement Types: Descriptions like 'Limb movement', 'General movement', categorized to indicate the nature of the activity observed.
Event Codes: Numeric or categorical codes that might relate to the clinical significance or characteristics of each movement.
Data Integration and Processing:
The integration of the sensor and annotation data is crucial for this tool, as it aligns the time-stamped sensor readings with the types of movements observed in the ultrasound. This allows for precise correlation and facilitates advanced analyses such as identifying patterns or predicting fetal health outcomes based on the sensor data.

Analytical Implications:
By structuring and correlating these datasets, the tool sets the stage for various analytical applications, from basic research in fetal development to advanced machine learning models designed to predict or classify fetal conditions. The processed data is particularly valuable for studies aiming to understand fetal behavior patterns or to develop non-invasive monitoring techniques that can be used in clinical settings.
