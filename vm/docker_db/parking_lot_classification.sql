--
-- PostgreSQL database dump
--

-- Dumped from database version 14.1
-- Dumped by pg_dump version 14.1

-- Started on 2021-12-16 14:52:15 MSK

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 3578 (class 1262 OID 16386)
-- Name: parking lot classification; Type: DATABASE; Schema: -; Owner: postgres
--

CREATE DATABASE "parking lot classification" WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE = 'en_US.UTF-8';


ALTER DATABASE "parking lot classification" OWNER TO postgres;

\connect -reuse-previous=on "dbname='parking lot classification'"

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 209 (class 1259 OID 16387)
-- Name: occupied_parking_lots; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.occupied_parking_lots (
    "DateTime" timestamp with time zone NOT NULL,
    "Place" smallint NOT NULL,
    "Camera_ID" smallint DEFAULT 0 NOT NULL,
    "Confidence" real
);


ALTER TABLE public.occupied_parking_lots OWNER TO postgres;

--
-- TOC entry 3572 (class 0 OID 16387)
-- Dependencies: 209
-- Data for Name: occupied_parking_lots; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 3432 (class 2606 OID 16397)
-- Name: occupied_parking_lots occupied_parking_lots_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.occupied_parking_lots
    ADD CONSTRAINT occupied_parking_lots_pkey PRIMARY KEY ("DateTime", "Camera_ID");


-- Completed on 2021-12-16 14:52:15 MSK

--
-- PostgreSQL database dump complete
--

