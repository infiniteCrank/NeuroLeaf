import { LanguageClassifier } from '../../src/tasks/LanguageClassifier';
import { ELMConfig } from '../../src/core/ELMConfig';

const input = document.getElementById('langInput') as HTMLInputElement;
const fill = document.getElementById('langFill') as HTMLDivElement;

const config: ELMConfig = {
    categories: ['English', 'French', 'Spanish'],
    hiddenUnits: 10,
    maxLen: 10,
    activation: 'relu',
    charSet: 'abcdefghijklmnopqrstuvwxyz',
    useTokenizer: false
};

const trainingData = [
    { text: 'hello', label: 'English' },
    { text: 'how are you', label: 'English' },
    { text: 'bonjour', label: 'French' },
    { text: 'comment ça va', label: 'French' },
    { text: 'hola', label: 'Spanish' },
    { text: 'como estas', label: 'Spanish' }
];

const classifier = new LanguageClassifier(config);
classifier.train(trainingData);

input.addEventListener('input', () => {
    const typed = input.value.trim();
    if (!typed) {
        fill.style.width = '0%';
        fill.textContent = '';
        fill.style.background = '#ccc';
        return;
    }

    const [result] = classifier.predict(typed, 1);
    const percent = Math.round(result.prob * 100);
    fill.style.width = `${percent}%`;
    fill.textContent = `${result.label} (${percent}%)`;

    // Gradient background based on label
    fill.style.background = {
        English: 'linear-gradient(to right, green, lime)',
        French: 'linear-gradient(to right, blue, cyan)',
        Spanish: 'linear-gradient(to right, red, orange)'
    }[result.label] || '#999';
});
