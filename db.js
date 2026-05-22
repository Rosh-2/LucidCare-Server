const Pool = require("pg").Pool;
require("dotenv").config();

console.log("DB Config:", {
  user: process.env.DB_USER,
  host: process.env.DB_HOST,
  database: process.env.DB_NAME,
  ssl: process.env.DB_HOST?.includes("supabase")
});

const pool = new Pool({
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  host: process.env.DB_HOST,
  port: process.env.DB_PORT,
  database: process.env.DB_NAME,
  ssl: process.env.DB_HOST?.includes("supabase") ? { rejectUnauthorized: false } : false
});

module.exports = pool;
