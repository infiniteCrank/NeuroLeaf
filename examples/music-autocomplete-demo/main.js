// main.js - Demo using NeuroLeaf via UMD build

const presets = {
    english: {
        charSet: 'abcdefghijklmnopqrstuvwxyz',
        useTokenizer: false,
        hiddenUnits: 120,
        maxLen: 15
    },
    russian: {
        charSet: 'абвгдеёжзийклмнопрстуфхцчшщьыъэюя',
        useTokenizer: false,
        hiddenUnits: 120,
        maxLen: 15
    },
    emoji: {
        charSet: '🎸🎹🎷🎧🎻',
        useTokenizer: false,
        hiddenUnits: 120,
        maxLen: 5
    }
};

const genreMap = {
    english: ["rock", "pop", "jazz", "hip hop", "electronic", "folk"],
    russian: ["рок", "поп", "джаз", "электроника", "народная"],
    emoji: ["🎸", "🎹", "🎷", "🎧", "🎻"]
};

let currentModel;

function setupELM(presetKey, activation, categories) {
    const config = {
        ...presets[presetKey],
        categories: categories,
        activation: activation
    };
    currentModel = new window.NeuroLeaf.ELM(config);
    if (config.categories.length > 0) currentModel.train();

    const input = document.getElementById("genre-input");
    const output = document.getElementById("suggestions");
    window.NeuroLeaf.bindAutocompleteUI({ model: currentModel, inputElement: input, outputElement: output });
}

async function preloadAndTrainFromJSON(url, activation) {
    try {
        const response = await fetch(url);
        const data = await response.json();

        const validLabels = [...new Set(
            data
                .filter(d => d && typeof d.label === 'string' && d.label.trim() !== '')
                .map(d => d.label.trim())
        )];

        if (!validLabels.length) throw new Error("No valid labels found in JSON");

        const config = {
            ...presets.english,
            categories: validLabels,
            activation: activation
        };
        currentModel = new window.NeuroLeaf.ELM(config);
        currentModel.train();

        const input = document.getElementById("genre-input");
        const output = document.getElementById("suggestions");
        window.NeuroLeaf.bindAutocompleteUI({ model: currentModel, inputElement: input, outputElement: output });

        if (validLabels.length < data.length) {
            console.warn(`⚠️ Skipped ${data.length - validLabels.length} invalid entries from JSON.`);
        }
    } catch (err) {
        console.error("Failed to preload JSON data:", err);
    }
}

function setupUI() {
    const presetButtons = document.querySelectorAll('[data-preset]');
    const activationSelect = document.getElementById('activation-select');
    const uploadInput = document.getElementById('data-upload');

    presetButtons.forEach(button => {
        button.addEventListener('click', () => {
            const presetKey = button.getAttribute('data-preset');
            const activation = activationSelect.value;
            setupELM(presetKey, activation, genreMap[presetKey]);
        });
    });

    activationSelect.addEventListener('change', () => {
        const active = document.querySelector('[data-preset].active');
        const presetKey = active?.getAttribute('data-preset') || 'english';
        setupELM(presetKey, activationSelect.value, genreMap[presetKey]);
    });

    uploadInput.addEventListener('change', async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const text = await file.text();
        const ext = file.name.split('.').pop();

        let data = [];
        try {
            if (ext === 'csv') {
                data = window.NeuroLeaf.IO.importCSV(text);
            } else if (ext === 'tsv') {
                data = window.NeuroLeaf.IO.importTSV(text);
            } else {
                data = window.NeuroLeaf.IO.importJSON(text);
            }
        } catch (err) {
            console.error("Failed to parse uploaded file:", err);
            return;
        }

        let labels = [];
        if (data.length && 'label' in data[0]) {
            labels = [...new Set(data.map(d => d.label).filter(label => label && label.trim() !== ''))];
        } else if (data.length && Object.keys(data[0]).length === 1) {
            // Assume single column label TSV with just genres
            const key = Object.keys(data[0])[0];
            labels = [...new Set(data.map(d => d[key]).filter(label => label && label.trim() !== ''))];
        } else {
            console.warn("No valid labels found in uploaded file.");
            return;
        }

        const config = {
            ...presets.english,
            categories: labels,
            activation: activationSelect.value
        };
        currentModel = new window.NeuroLeaf.ELM(config);
        currentModel.train();

        const input = document.getElementById("genre-input");
        const output = document.getElementById("suggestions");
        window.NeuroLeaf.bindAutocompleteUI({ model: currentModel, inputElement: input, outputElement: output });
    });
}

// Boot
preloadAndTrainFromJSON("/music_genres.json", "relu");
setupUI();
