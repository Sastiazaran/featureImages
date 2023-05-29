const express = require('express');
const app = express();
const port = 3000
const fileUpload = require('express-fileupload');
const cors = require('cors');

app.use(fileUpload());
app.use(cors());
app.use(express.json());

app.get('/', (req, res) => {
    res.send("Hello World");
});

app.post('/upload', (req, res) => {
  console.log(req.files.file);
  res.send(`Archivo ${req.files.file.name} uploaded correcly`);

  //fileUpload.mv("./fotosastibichi");
});

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
})