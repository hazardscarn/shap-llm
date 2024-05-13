# shap-llm
Module to create Actional SHAP explanation and action recommendation using LLM and SHAP explainers from model created


POC created with Telco Churn data from Kaggle.

Refer to scripts in pyscripts for different approaches used:

Script 0 : Uses OpenAI model for local explanation for churn prediction and action recommendation
Script 1 : Uses Llama3 to create locally set Agents that create reports and action from global SHAP explanation based on question. Uses Corrective RAG technique here
Script 2 : Uses Llama3 to create locally set Agents that create reports and action from global SHAP explanation based on question. Uses SQL retriver and simple RAG here


Module to create SHAP summary for data is described in xgb_process folder
