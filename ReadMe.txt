# Bitcoin Price Prediction with LSTM

This project predicts Bitcoin prices using a Long Short-Term Memory (LSTM) model. It consists of a Flask web app that serves the trained model and provides predictions based on the latest Bitcoin price data.

## Features
- Train an LSTM model on historical Bitcoin data.
- Predict future Bitcoin prices based on the latest data.
- Provide buy, sell, or hold recommendations.
- Display a graphical trend of Bitcoin prices.

## Project Structure
```
├── app.py                     # Flask web application
├── train_model.py             # Script to train the LSTM model
├── bitcoin_price_lstm_model.h5 # Trained LSTM model
├── scaler.pkl                 # Scaler for data normalization
├── templates/
│   ├── index.html             # Frontend template (to be created)
├── static/
│   ├── bitcoin_price_graph.png # Generated price trend graph
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies (to be created)
```

## Installation
### **1. Clone the Repository**
```sh
git clone https://github.com/yourusername/bitcoin-price-prediction.git
cd bitcoin-price-prediction
```

### **2. Set Up a Virtual Environment** (Optional but recommended)
```sh
python -m venv venv  # Create virtual environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate  # Windows
```

### **3. Install Dependencies**
Create a `requirements.txt` file and include:
```txt
Flask
yfinance
tensorflow
numpy
pandas
scikit-learn
matplotlib
joblib
```
Then install dependencies:
```sh
pip install -r requirements.txt
```

## Training the Model
To train the LSTM model from scratch:
```sh
python train_model.py
```
This will fetch Bitcoin price data, train the model, and save `bitcoin_price_lstm_model.h5` and `scaler.pkl`.

## Running the Web App
After training the model, run the Flask app:
```sh
python app.py
```
Then open `http://127.0.0.1:5000/` in your browser.

## API Usage
### **Predict Bitcoin Price**
**Endpoint:** `POST /predict`

**Response:**
```json
{
    "predicted_price": 50000.0,
    "latest_price": 49000.0,
    "high_price": 49500.0,
    "low_price": 48500.0,
    "suggestion": "Buy",
    "graph": "static/bitcoin_price_graph.png"
}
```

## Future Improvements
- Add a front-end UI.
- Deploy to a cloud server (e.g., AWS, Heroku).
- Improve model accuracy with more features.

## License
MIT License. Feel free to modify and use it.

