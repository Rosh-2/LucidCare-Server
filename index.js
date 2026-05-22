const express = require("express");
const app = express();
const cors = require("cors");

// middleware
app.use(express.json()); // req.body
app.use(cors());

// routes
app.use("/auth", require("./routes/jwtAuth"));

const PORT = 5000;

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
