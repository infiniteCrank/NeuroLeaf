const {
    ModulePool,
    ELM,
    MiniTransformer,
    SignalBus,
    TFIDFVectorizer
} = window.astermind;

const pool = new ModulePool();
const bus = new SignalBus();
const defaultCorpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "Cybernetic organisms adapt and learn from signals.",
    "Extreme Learning Machines are efficient and powerful.",
    "Signal buses distribute input across connected modules."
];
const vectorizer = new TFIDFVectorizer(defaultCorpus);

// Create 500 modules
for (let i = 0; i < 500; i++) {
    if (i % 2 === 0) {
        const elm = new ELM({
            categories: ["default"],
            hiddenUnits: 20,
            maxLen: 40,
            activation: "relu",
            dropout: 0.1,
            charSet: "abcdefghijklmnopqrstuvwxyz ",
            useTokenizer: false,
            log: { verbose: false }
        });
        pool.addModule({
            id: `mod-${i}`,
            type: "ELM",
            elm,
            role: "listener",
            metrics: { recall1: 0, recall5: 0, mrr: 0, avgLatency: 0 },
            lastEvaluated: Date.now(),
            signalLikes: 0,
            signalDislikes: 0,
            signalsConsumedByOthers: 0,
            emittedSignalIds: [],
            signalHistory: []
        });
    } else {
        const vocab = new window.astermind.Vocab("abcdefghijklmnopqrstuvwxyz ".split(""));
        const transformer = new MiniTransformer(vocab);
        pool.addModule({
            id: `mod-${i}`,
            type: "Transformer",
            transformer,
            role: "explorer",
            metrics: { recall1: 0, recall5: 0, mrr: 0, avgLatency: 0 },
            lastEvaluated: Date.now(),
            signalLikes: 0,
            signalDislikes: 0,
            signalsConsumedByOthers: 0,
            emittedSignalIds: [],
            signalHistory: []
        });
    }
}

const canvas = document.getElementById("cyberCanvas");
const ctx = canvas.getContext("2d");
canvas.width = 1200;
canvas.height = 800;

const timestampMap = {};
const modulePositions = {};

pool.listModules().forEach((mod, index) => {
    const cols = Math.floor(canvas.width / 25);
    const x = 25 * (index % cols) + 5;
    const y = 25 * Math.floor(index / cols) + 5;
    modulePositions[mod.id] = { x, y };
});

function drawModules() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    pool.listModules().forEach(mod => {
        const { x, y } = modulePositions[mod.id];
        const active = timestampMap[mod.id] && Date.now() - timestampMap[mod.id] < 3000;
        ctx.fillStyle = active ? "#00ffcc" : "#333";
        ctx.fillRect(x, y, 15, 15);
        ctx.strokeStyle = "#ff00ff";
        ctx.strokeRect(x, y, 15, 15);
    });
}

// Handle text input
document.getElementById("userInput").addEventListener("input", (e) => {
    const text = e.target.value.trim();
    if (!text) return;

    const vec = vectorizer.vectorize(text);
    pool.listModules().forEach(mod => {
        timestampMap[mod.id] = Date.now();
        bus.broadcast({
            id: `${mod.id}-human-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
            sourceModuleId: mod.id,
            vector: vec,
            timestamp: Date.now(),
            metadata: { role: "human", parents: [] }
        });
    });

    drawModules();
});

// Handle mouse click
canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    pool.listModules().forEach(mod => {
        const { x, y } = modulePositions[mod.id];
        if (
            clickX >= x && clickX <= x + 15 &&
            clickY >= y && clickY <= y + 15
        ) {
            timestampMap[mod.id] = Date.now();
            const vec = Array.from({ length: vectorizer.vocabulary.length }, () => Math.random() * 2 - 1);
            bus.broadcast({
                id: `${mod.id}-touch-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
                sourceModuleId: mod.id,
                vector: vec,
                timestamp: Date.now(),
                metadata: { role: "synthetic", parents: [] }
            });
            console.log(`🖐️ You touched ${mod.id}`);
        }
    });

    drawModules();
});

// Periodic loop
setInterval(() => {
    pool.broadcastAllSignals(bus, vectorizer);
    pool.consumeAllSignals(bus);
    pool.evolveModules(new window.astermind.FeedbackStore());
    drawModules();

    pool.listModules().forEach(mod => {
        if (mod.elm && Math.random() < 0.02) {
            mod.elm.saveModelAsJSONFile(`${mod.id}-weights.json`);
            console.log(`💾 ${mod.id} saved its weights.`);
        }
        if (Math.random() < 0.01) {
            const msg = `Module ${mod.id} says hello (${new Date().toLocaleTimeString()})`;
            const output = document.getElementById("organismOutput");
            if (output) {
                const div = document.createElement("div");
                div.textContent = msg;
                output.appendChild(div);
            }
        }
    });
}, 5000);
