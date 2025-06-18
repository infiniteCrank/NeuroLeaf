// @ts-ignore
const { LanguageClassifier } = window.NeuroLeaf;

const input = document.getElementById('langInput') as HTMLInputElement;
const fill = document.getElementById('langFill') as HTMLDivElement;

const charSet = 'abcdefghijklmnopqrstuvwxyzçàéèñáéíóúü¿¡ ';
const config = {
    categories: ['English', 'French', 'Spanish'],
    hiddenUnits: 50,
    maxLen: 30,
    activation: 'relu',
    charSet,
    useTokenizer: true,
    tokenizerDelimiter: /\s+/,
};

fetch('/language_greetings_600.csv')
    .then(res => res.text())
    .then(csv => {
        const lines = csv
            .split('\n')
            .map(l => l.trim())
            .filter(Boolean)
            .slice(1); // Skip header

        const trainingData = lines.map(line => {
            const [text = '', label = ''] = line.split(',');
            return {
                text: text.trim().toLowerCase(), // Normalize casing
                label: label.trim()
            };
        }).filter(d => d.text && d.label);

        const classifier = new LanguageClassifier(config);
        classifier.train(trainingData);

        input.addEventListener('input', () => {
            const typed = input.value.trim().toLowerCase();
            if (!typed) {
                fill.style.width = '0%';
                fill.textContent = '';
                fill.style.background = '#ccc';
                return;
            }

            const [result] = classifier.predict(typed, 1);
            const percent = Math.round(result.prob * 100);
            fill.style.width = `${percent}%`;

            fill.textContent = percent < 40
                ? '🤔 Not sure'
                : `${result.label} (${percent}%)`;

            fill.style.background = {
                English: 'linear-gradient(to right, green, lime)',
                French: 'linear-gradient(to right, blue, cyan)',
                Spanish: 'linear-gradient(to right, red, orange)'
            }[result.label] || '#999';
        });
    });
