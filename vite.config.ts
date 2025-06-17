// vite.config.ts
import { defineConfig } from 'vite';

export default defineConfig(({ command }) => {
    const isDemo = command === 'serve' && process.cwd().includes('examples');

    return {
        root: isDemo ? 'examples/music-autocomplete-demo' : '.',
        publicDir: isDemo ? '../../public' : 'public',
        server: {
            port: 5173,
        },
        test: {
            include: ['tests/**/*.test.ts'],
        },
    };
});
