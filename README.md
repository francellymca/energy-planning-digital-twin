# Energy Planning Digital Twin

> **Data-driven framework for energy demand forecasting, scenario analysis, and decision support in regional energy planning.**

## Overview

The **Energy Planning Digital Twin** is an academic and applied research project focused on the development of a computational framework to support energy planning in Brazilian states.

The project integrates public energy, socioeconomic, and potentially climate-related datasets with **energy demand forecasting, scenario analysis, Explainable Artificial Intelligence (XAI), and Digital Twin concepts**.

The proposed framework aims to investigate how different regional characteristics influence electricity demand trajectories and how data-driven models can support long-term energy planning and decision-making.

> **Status:** In Development
> **Research area:** Energy and Resource Planning
> **Current phase:** Research design and data feasibility assessment

---

## Research Question

**How can electricity demand forecasting models integrated with scenario analysis support energy planning in Brazilian states with different energy, socioeconomic, and climate characteristics?**

---

## Research Objective

Develop a computational framework for energy planning that integrates **electricity demand forecasting and scenario analysis** across Brazilian states with different energy, socioeconomic, and climate characteristics.

---

## Initial Scope

The study will initially investigate a selected group of Brazilian states representing different regional and energy characteristics.

Initial candidates include:

* São Paulo (SP)
* Minas Gerais (MG)
* Rio de Janeiro (RJ)
* Paraná (PR)
* Bahia (BA)
* Rio Grande do Sul (RS)

The final selection will depend on **data availability, temporal coverage, consistency, and relevance to the proposed analysis**.

Rio Grande do Sul may also be investigated as a case study for assessing energy-system behavior and resilience under extreme climate events, depending on data feasibility.

---

## Proposed Architecture

```text
Public Data Sources
        │
        ▼
Data Collection
        │
        ▼
Data Processing & Integration
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Energy Demand Forecasting
        │
        ▼
Explainable AI (XAI)
        │
        ▼
Scenario Analysis
        │
        ▼
Simplified Digital Twin
        │
        ▼
Indicators & Dashboard
        │
        ▼
Energy Planning & Decision Support
```

---

## Planned Data Sources

Public datasets will be evaluated from institutions such as:

* EPE — Empresa de Pesquisa Energética
* ONS — Operador Nacional do Sistema Elétrico
* ANEEL — Agência Nacional de Energia Elétrica
* IBGE — Instituto Brasileiro de Geografia e Estatística
* Public climate and meteorological databases

The final datasets and variables will be defined after the **data feasibility assessment**.

---

## Repository Structure

```text
energy-planning-digital-twin/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── dashboard/
├── docs/
├── notebooks/
├── references/
├── src/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Planned Development

The project will be developed incrementally through the following stages:

**Research Design → Data Assessment → Data Engineering → Exploratory Analysis → Forecasting → Model Validation → XAI → Scenario Analysis → Digital Twin → Decision Support**

---

## Academic Context

This project is being developed within the **Energy and Resource Planning** course at **São Paulo State University (UNESP)**.

The research combines concepts from **Electrical Engineering, Energy Planning, Data Science, Machine Learning, and Automation**.

---

## Technologies

Technologies will be selected according to the methodological requirements of the research.

Expected tools include:

`Python` • `Pandas` • `Scikit-learn` • `Time Series Forecasting` • `Machine Learning` • `XAI` • `Data Visualization`

Additional technologies may be incorporated as the project evolves.

---

## Project Status

🚧 **In Development**

Current activities:

* [x] Research topic definition
* [x] Research question definition
* [x] Initial project architecture
* [x] Repository structure
* [ ] Public data feasibility assessment
* [ ] Dataset definition
* [ ] Exploratory data analysis
* [ ] Forecasting models
* [ ] Model validation
* [ ] Explainable AI analysis
* [ ] Scenario development
* [ ] Simplified Digital Twin
* [ ] Dashboard and decision-support layer
* [ ] Final research paper