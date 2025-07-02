import fs from "fs";
import { PCA } from "ml-pca";
import { Matrix } from "ml-matrix";

interface EmbeddingRecord {
    embedding: number[];
    metadata: {
        text: string;
    };
}

// Load embeddings
const data: EmbeddingRecord[] = JSON.parse(
    fs.readFileSync("/Users/julianwilkison-duran/Documents/NeuroLeaf/node_examples/embeddings/combined_embeddings.json", "utf8")
);
console.log(`✅ Loaded ${data.length} embeddings.`);

// Prepare matrix
const vectors = data.map(d => d.embedding);
const m = new Matrix(vectors);

// Run PCA
const pca = new PCA(m);
const reduced = pca.predict(m, { nComponents: 2 }).to2DArray();

console.log(`✅ PCA completed.`);

// Write CSV
const csvLines = [
    "x,y,text",
    ...reduced.map((v, i) => `${v[0]},${v[1]},"${data[i].metadata.text.replace(/"/g, '""')}"`)
];
fs.writeFileSync("./embeddings_pca.csv", csvLines.join("\n"));
console.log(`💾 Saved CSV to embeddings_pca.csv.`);

// Generate HTML visualization
const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Embedding PCA Visualization</title>
  <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist@2.24.1/plotly.min.js"></script>
</head>
<body>
  <h1>Embedding Visualization (PCA)</h1>
  <div id="plot" style="width: 100%; height: 90vh;"></div>
  <script>
    const points = ${JSON.stringify(reduced)};
    const texts = ${JSON.stringify(data.map(d => d.metadata.text.slice(0, 100)))};

    const trace = {
      x: points.map(p => p[0]),
      y: points.map(p => p[1]),
      mode: 'markers',
      type: 'scatter',
      text: texts,
      marker: { size: 6 }
    };

    Plotly.newPlot('plot', [trace], {
      title: 'Embeddings PCA Projection',
      hovermode: 'closest'
    });
  </script>
</body>
</html>
`;
fs.writeFileSync("./embeddings_pca.html", html);
console.log(`💾 Saved HTML visualization to embeddings_pca.html.`);
