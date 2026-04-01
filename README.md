I’m excited to share my recent project on an AI-Driven Crop Recommendation System, designed to support smarter and more sustainable farming decisions.

This system analyzes key soil and climate parameters such as:

<img width="634" height="508" alt="Screenshot 2026-03-23 174212" src="https://github.com/user-attachments/assets/4e26f735-5e77-406d-804a-a22cdc244b96" />

Nitrogen (N), Phosphorus (P), Potassium (K)
Soil pH level
Temperature, Humidity
Rainfall

Based on these inputs, the model provides accurate crop recommendations, helping farmers choose the most suitable crops for their land conditions.


<img width="229" height="454" alt="Screenshot 2026-03-23 180411" src="https://github.com/user-attachments/assets/eae7dbf9-6c44-4777-a44d-6c014c6d6376" />

The system follows a structured workflow:
The  project follows a systematic and well-defined workflow that transforms raw agricultural data into meaningful crop recommendations through multiple stages of processing and analysis. The process begins with the collection of input data from the user, which includes essential soil parameters such as Nitrogen (N), Phosphorus (P), Potassium (K), and pH value, along with important climatic conditions like temperature, humidity, and rainfall. These inputs are carefully validated to ensure they fall within acceptable ranges for accurate prediction. Once the data is collected, it undergoes a feature engineering phase, where additional derived features such as soil health score, nutrient balance index, and climate suitability index are computed to enhance the quality and representation of the data. This enriched dataset is then passed to the model processing stage, where multiple machine learning algorithms including XGBoost, LightGBM, Random Forest, and ExtraTrees are applied simultaneously as part of an ensemble learning approach. Each model independently analyzes the input data and generates predictions, which are then combined using a voting mechanism to produce a final, highly accurate crop recommendation. After the prediction is generated, the system employs SHAP (SHapley Additive exPlanations) to interpret the results by identifying and quantifying the contribution of each input feature toward the final decision. This step ensures transparency and helps users understand why a particular crop was recommended. Finally, the output is presented to the user through an intuitive interface, where the recommended crop is displayed along with insights into the influencing factors, enabling informed decision-making. This structured workflow ensures efficiency, accuracy, and reliability in the crop recommendation process while making the system accessible and understandable for end users..




💡 What makes this project unique?

Integrates both soil nutrients and real-time climate factors
Focuses on data-driven decision making
Aims to improve crop yield and reduce risk
Built with a user-friendly interface for easy input and analysis

🚀 Key Features:
✔ Intelligent crop prediction
✔ Simple and interactive UI
✔ Preset crop testing (Rice, Maize, Mango, Cotton, Watermelon)
✔ Practical application for precision agriculture

📊 This project reflects my interest in Data Science, Machine Learning, and real-world problem solving, especially in the agriculture domain.




<img width="1472" height="742" alt="Screenshot 2026-03-26 172043" src="https://github.com/user-attachments/assets/958eb60a-74bc-4386-ae6e-764f55a2eee4" />


