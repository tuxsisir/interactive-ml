-- Database: mlflask
-- DROP DATABASE mlflask;

CREATE DATABASE mlflask
    WITH 
    OWNER = macq
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;	

DROP TABLE if exists ML_USER;
CREATE TABLE ML_USER(
	created_at timestamp NULL,
	user_id INT GENERATED ALWAYS AS IDENTITY,
	first_name VARCHAR(255) NOT NULL,
	last_name VARCHAR(255) NOT NULL,
	phone VARCHAR(15),
	email VARCHAR(100),
	twitter_handle VARCHAR(255),
	PRIMARY KEY(user_id)
);

DROP TABLE if exists ML_PROJECT;
CREATE TABLE ML_PROJECT(
	project_id INT GENERATED ALWAYS AS IDENTITY,
	created_by INT,
	title VARCHAR(255) NOT NULL,
	dataset VARCHAR(255) NOT NULL,
	description TEXT,
	PRIMARY KEY(project_id),
	CONSTRAINT fk_created_by
      FOREIGN KEY(created_by) 
	  REFERENCES ML_USER(user_id)
);

DROP TABLE if exists ML_PROJECT_CONFIG;
CREATE TABLE ML_PROJECT_CONFIG(
	config_id INT GENERATED ALWAYS AS IDENTITY,
	created_by INT,
	project_id INT,
	description TEXT,
	project_info json not null,
	PRIMARY KEY(config_id),
	CONSTRAINT fk_created_by FOREIGN KEY(created_by) REFERENCES ML_USER(user_id),
	CONSTRAINT fk_project_id FOREIGN KEY(project_id) REFERENCES ML_PROJECT(project_id)
);


