import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, TextInput, ScrollView } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import * as Crypto from 'crypto-js';
import { SafeAreaView } from 'react-native-safe-area-context';

const GRID_SIZE = 5;
const TILE_COUNT = GRID_SIZE * GRID_SIZE;
const DEFAULT_MINE_COUNT = 3;
const TRAINING_SAMPLES = 500;
const MONTE_CARLO_TRIALS = 10000;

function generateMines(seed, nonce, mineCount = DEFAULT_MINE_COUNT) {
  const input = `${seed}-${nonce}`;
  const hash = Crypto.SHA256(input).toString();
  const numbers = [];

  for (let i = 0; i < hash.length; i += 4) {
    const segment = hash.substr(i, 4);
    const num = parseInt(segment, 16) % TILE_COUNT;
    if (!numbers.includes(num)) {
      numbers.push(num);
      if (numbers.length === mineCount) break;
    }
  }

  return numbers;
}

function monteCarloPredict(mineCount, selectedIndexes = []) {
  const safeCount = Array(TILE_COUNT).fill(0);

  for (let trial = 0; trial < MONTE_CARLO_TRIALS; trial++) {
    const tiles = [...Array(TILE_COUNT).keys()];
    const mines = [];

    while (mines.length < mineCount) {
      const randIndex = Math.floor(Math.random() * tiles.length);
      const pick = tiles.splice(randIndex, 1)[0];
      mines.push(pick);
    }

    for (let i = 0; i < TILE_COUNT; i++) {
      if (!mines.includes(i) && !selectedIndexes.includes(i)) {
        safeCount[i] += 1;
      }
    }
  }

  return safeCount.map((count) => count / MONTE_CARLO_TRIALS);
}

async function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [TILE_COUNT], units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: TILE_COUNT, activation: 'sigmoid' }));
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
  return model;
}

async function trainModel(model) {
  const inputs = [];
  const outputs = [];

  for (let i = 0; i < TRAINING_SAMPLES; i++) {
    const selected = Array(TILE_COUNT).fill(0);
    const safeTiles = Array(TILE_COUNT).fill(0);

    const count = Math.floor(Math.random() * 5);
    const indexes = [];
    while (indexes.length < count) {
      const rand = Math.floor(Math.random() * TILE_COUNT);
      if (!indexes.includes(rand)) indexes.push(rand);
    }
    indexes.forEach((idx) => (selected[idx] = 1));

    const probs = monteCarloPredict(DEFAULT_MINE_COUNT, indexes);
    inputs.push(selected);
    outputs.push(probs);
  }

  const xs = tf.tensor2d(inputs);
  const ys = tf.tensor2d(outputs);

  await model.fit(xs, ys, {
    epochs: 10,
    batchSize: 32,
  });
}

export default function App() {
  const [selected, setSelected] = useState(Array(TILE_COUNT).fill(false));
  const [seed, setSeed] = useState('');
  const [nonce, setNonce] = useState('');
  const [mines, setMines] = useState([]);
  const [safetyMap, setSafetyMap] = useState(Array(TILE_COUNT).fill(0));
  const [model, setModel] = useState(null);

  useEffect(() => {
    const init = async () => {
      await tf.ready();
      const m = await createModel();
      await trainModel(m);
      setModel(m);
    };
    init();
  }, []);

  const handlePress = (index) => {
    const updated = [...selected];
    updated[index] = !updated[index];
    setSelected(updated);
  };

  const verifySeed = () => {
    const mineIndexes = generateMines(seed, nonce);
    setMines(mineIndexes);
  };

  const runMonteCarlo = () => {
    const selectedIndexes = selected.map((val, idx) => (val ? idx : -1)).filter((i) => i !== -1);
    const prediction = monteCarloPredict(DEFAULT_MINE_COUNT, selectedIndexes);
    setSafetyMap(prediction);
  };

  const runNeuralNet = async () => {
    if (!model) return;
    const input = tf.tensor([selected.map((val) => (val ? 1 : 0))]);
    const output = model.predict(input);
    const prediction = await output.data();
    setSafetyMap(Array.from(prediction));
  };

  const renderGrid = () => {
    const tiles = [];
    for (let i = 0; i < TILE_COUNT; i++) {
      const isMine = mines.includes(i);
      const probability = safetyMap[i];
      const bgColor = isMine
        ? '#ef4444'
        : selected[i]
        ? '#4ade80'
        : `rgba(96,165,250,${probability})`;

      tiles.push(
        <TouchableOpacity
          key={i}
          onPress={() => handlePress(i)}
          style={{
            width: 60,
            height: 60,
            margin: 4,
            backgroundColor: bgColor,
            justifyContent: 'center',
            alignItems: 'center',
            borderRadius: 12,
          }}
        >
          <Text style={{ color: 'white', fontSize: 14 }}>
            {isMine ? '💣' : selected[i] ? '✅' : `${Math.round(probability * 100)}%`}
          </Text>
        </TouchableOpacity>
      );
    }
    return tiles;
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#111827' }}>
      <ScrollView contentContainerStyle={{ alignItems: 'center', justifyContent: 'center', paddingTop: 30 }}>
        <Text style={{ color: 'white', fontSize: 24, marginBottom: 20 }}>Stake-style Mines (5x5)</Text>

        <View style={{ marginBottom: 20 }}>
          <TextInput
            placeholder='Enter Seed'
            placeholderTextColor='#9ca3af'
            value={seed}
            onChangeText={setSeed}
            style={{ backgroundColor: '#1f2937', color: 'white', padding: 10, borderRadius: 8, marginBottom: 10, width: 250 }}
          />
          <TextInput
            placeholder='Enter Nonce'
            placeholderTextColor='#9ca3af'
            value={nonce}
            onChangeText={setNonce}
            style={{ backgroundColor: '#1f2937', color: 'white', padding: 10, borderRadius: 8, width: 250 }}
          />
          <TouchableOpacity onPress={verifySeed} style={{ backgroundColor: '#2563eb', padding: 10, borderRadius: 8, marginTop: 10 }}>
            <Text style={{ color: 'white' }}>Verify Seed</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={runMonteCarlo} style={{ backgroundColor: '#10b981', padding: 10, borderRadius: 8, marginTop: 10 }}>
            <Text style={{ color: 'white' }}>Run Monte Carlo</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={runNeuralNet} style={{ backgroundColor: '#f59e0b', padding: 10, borderRadius: 8, marginTop: 10 }}>
            <Text style={{ color: 'white' }}>Run Neural Net</Text>
          </TouchableOpacity>
        </View>

        <View style={{ width: '90%', flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center' }}>
          {renderGrid()}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}