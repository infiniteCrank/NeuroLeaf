import { defineConfig } from 'vite';

export default defineConfig({
    root: 'examples/music-autocomplete-demo',
    publicDir: '../../public',
    server: {
        port: 5173,
    },
});
