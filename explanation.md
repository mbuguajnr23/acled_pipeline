# Conflict Event Prediction in Africa (2012-2022) using ACLED Dataset

This project aims to analyze trends in political violence and conflict events across African countries over a 10-year period (2012-2022) using the Armed Conflict Location & Event Data Project (ACLED) dataset. The ultimate goal is to develop a predictive model capable of forecasting the likelihood of future conflict-related events in specific regions.



## Table of Contents
1.  [Project Goal](#project-goal)
2.  [Dataset](#dataset)
3.  [Project Structure](#project-structure)
4.  [Setup and Installation](#setup-and-installation)
5.  [Workflow](#workflow)
    *   [5.1 Data Acquisition](#51-data-acquisition)
    *   [5.2 Exploratory Data Analysis (EDA)](#52-exploratory-data-analysis-eda)
    *   [5.3 Feature Engineering](#53-feature-engineering)
6.  [Key Scripts](#key-scripts)
7.  [Next Steps](#next-steps)

## 1. Project Goal
*   To identify patterns and trends of conflict over time and space in Africa using the ACLED dataset for the period 2012-2022.
*   To develop a machine learning model to forecast the likelihood of conflict events occurring in a given African region (Admin1 level) in the near future (e.g., the following month).
*   To identify which features (e.g., past event types, actor groups, fatalities, regional characteristics) are most predictive of future conflict.

## 2. Dataset
The primary dataset used is from the **Armed Conflict Location & Event Data Project (ACLED)**.
*   **Source:** [ACLED Website](https://acleddata.com/)
*   **Description:** ACLED is a disaggregated data collection, analysis, and crisis mapping project. It collects the dates, actors, locations, fatalities, and types of all reported political violence and protest events around the world.
*   **Coverage for this project:** Africa, events from 2012-01-01 to 2022-12-31.
*   **Key Data Fields Used:** `event_date`, `event_type`, `sub_event_type`, `actor1`, `inter1` (actor type), `country`, `admin1` (primary subnational administrative unit), `fatalities`, `latitude`, `longitude`.

## 3. Project Structure

