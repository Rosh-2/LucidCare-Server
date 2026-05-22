CREATE DATABASE lucidcare;

-- extension for uuid
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE users(
  user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_fullname VARCHAR(255) NOT NULL,
  user_email VARCHAR(255) NOT NULL,
  user_password VARCHAR(255) NOT NULL,
  user_age INTEGER,
  user_sex VARCHAR(50)
);

CREATE TABLE summaries(
  summary_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES users(user_id) NOT NULL,
  summary_text TEXT NOT NULL,
  language VARCHAR(50) DEFAULT 'English',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
