# Stake Predictor (Expo + Termux Ready)

## 📦 Features
- Stake-style Mines Predictor (5x5 grid)
- ✅ Seed Verifier (SHA-256)
- 🔁 Monte Carlo Simulation
- 🧠 TensorFlow.js Neural Network Training

## 📱 How to Run on Termux

```bash
pkg update && pkg install nodejs git
npm install -g expo-cli eas-cli
git clone https://github.com/yourname/stake-mines-ai-expo.git
cd stake-mines-ai-expo
npm install
expo start --tunnel
```

Then scan the QR using **Expo Go**.

## 📤 To Build APK

```bash
npx eas build --platform android --profile preview
```

You will receive a downloadable `.apk` link.