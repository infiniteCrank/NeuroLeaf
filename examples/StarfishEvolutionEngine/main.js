const {
    ModulePool
} = window.astermind;

document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("userInput");

    input.addEventListener("input", () => {
        console.log("Typing:", input.value);
    });
});