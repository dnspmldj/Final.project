{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/dnspmldj/Final.project/blob/main/GROUP3_FINAL_PROJECT.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# GROUP 3\n",
        "# FINAL PROJECT IN EMERGING TECHNOLOGIES 2: FASHION DATASET\n",
        "## Members: <BR>\n",
        "### **Doroteo, Victor Ponce C.**\n",
        "### **Echiverri, Syd**\n",
        "### **De Leon, Sheina Mae**\n",
        "### **De Jose, Dennisse**\n",
        "### **Domondon, Mark Stefan**\n",
        "### INSTRUCTOR:\n",
        "### **DR. JONATHAN TAYLAR**<BR>\n",
        "## COURSE/SECTION:\n",
        "### **CPE019/CPE32S1**"
      ],
      "metadata": {
        "id": "K4TwNzieVc7b"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **IMPORTING MODULES:**"
      ],
      "metadata": {
        "id": "OE7weVD2YEDy"
      }
    },
    {
      "cell_type": "code",
      
      "metadata": {
        "id": "Po_BBGfUVYHo"
      },
      "outputs": [],
      "source": [
        "import streamlit as st",
        "import time\n",
        "import numpy as np\n",
        "import sys\n",
        "from matplotlib import pyplot\n",
        "import pandas as pd\n",
        "from PIL import Image\n",
        "from numpy import mean\n",
        "from numpy import std\n",
        "from numpy import argmax\n",
        "from matplotlib import pyplot as plt\n",
        "from sklearn.model_selection import KFold\n",
        "from keras.preprocessing.image import ImageDataGenerator\n",
        "from keras.models import Sequential\n",
        "from keras.layers import Dropout\n",
        "from keras.layers import BatchNormalization\n",
        "from keras.layers import Conv2D\n",
        "from keras.layers import MaxPooling2D\n",
        "from keras.layers import Dense\n",
        "from keras.layers import Flatten\n",
        "from keras.optimizers import SGD\n",
        "from keras.models import load_model\n",
        "from keras.utils import load_img\n",
        "from keras.utils import img_to_array\n",
        "from keras.utils import to_categorical"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Fashion MNIST Dataset - It contains images of different outfits or those to wear such as shoes, t-shirts, dresses, trousers, boots and many more. It should be able to predict what kind it is when given a test image.**"
      ],
      "metadata": {
        "id": "wGOGI7rTYsUz"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **IMPORTING CSV FILE**"
      ],
      "metadata": {
        "id": "Ujr01x2LY2dE"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train_dataset = pd.read_csv(\"/content/drive/MyDrive/Colab Notebooks/fashion-mnist_train.csv\")\n",
        "test_dataset = pd.read_csv(\"/content/drive/MyDrive/Colab Notebooks/fashion-mnist_test.csv\")"
      ],
      "metadata": {
        "id": "S040Nj9jY0Kf"
      },
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_dataset.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0p0HLbG9ZwlF",
        "outputId": "78f13238-b799-4b61-f81e-3c1854317ee9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(60000, 785)"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "test_dataset.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GYViEqlMZ0fl",
        "outputId": "230ce37f-da11-4635-c87d-c1435a55036b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(10000, 785)"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_dataset.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 236
        },
        "id": "-778frDbZ1-r",
        "outputId": "a2e58c2f-1c2c-4208-fbee-3c099d120e7a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   label  pixel1  pixel2  pixel3  pixel4  pixel5  pixel6  pixel7  pixel8  \\\n",
              "0      2       0       0       0       0       0       0       0       0   \n",
              "1      9       0       0       0       0       0       0       0       0   \n",
              "2      6       0       0       0       0       0       0       0       5   \n",
              "3      0       0       0       0       1       2       0       0       0   \n",
              "4      3       0       0       0       0       0       0       0       0   \n",
              "\n",
              "   pixel9  ...  pixel775  pixel776  pixel777  pixel778  pixel779  pixel780  \\\n",
              "0       0  ...         0         0         0         0         0         0   \n",
              "1       0  ...         0         0         0         0         0         0   \n",
              "2       0  ...         0         0         0        30        43         0   \n",
              "3       0  ...         3         0         0         0         0         1   \n",
              "4       0  ...         0         0         0         0         0         0   \n",
              "\n",
              "   pixel781  pixel782  pixel783  pixel784  \n",
              "0         0         0         0         0  \n",
              "1         0         0         0         0  \n",
              "2         0         0         0         0  \n",
              "3         0         0         0         0  \n",
              "4         0         0         0         0  \n",
              "\n",
              "[5 rows x 785 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-17e603cf-d5dd-43ef-84c7-3b88f0ced1c0\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>label</th>\n",
              "      <th>pixel1</th>\n",
              "      <th>pixel2</th>\n",
              "      <th>pixel3</th>\n",
              "      <th>pixel4</th>\n",
              "      <th>pixel5</th>\n",
              "      <th>pixel6</th>\n",
              "      <th>pixel7</th>\n",
              "      <th>pixel8</th>\n",
              "      <th>pixel9</th>\n",
              "      <th>...</th>\n",
              "      <th>pixel775</th>\n",
              "      <th>pixel776</th>\n",
              "      <th>pixel777</th>\n",
              "      <th>pixel778</th>\n",
              "      <th>pixel779</th>\n",
              "      <th>pixel780</th>\n",
              "      <th>pixel781</th>\n",
              "      <th>pixel782</th>\n",
              "      <th>pixel783</th>\n",
              "      <th>pixel784</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>9</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>6</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>30</td>\n",
              "      <td>43</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 785 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-17e603cf-d5dd-43ef-84c7-3b88f0ced1c0')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-17e603cf-d5dd-43ef-84c7-3b88f0ced1c0 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-17e603cf-d5dd-43ef-84c7-3b88f0ced1c0');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-53ef4cf0-1118-4691-a363-4923c8792ff0\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-53ef4cf0-1118-4691-a363-4923c8792ff0')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-53ef4cf0-1118-4691-a363-4923c8792ff0 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **FORMATTING THE TRAIN AND TEST SETS**"
      ],
      "metadata": {
        "id": "DrzgVCSCZ42z"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "img_rows, img_cols = 28, 28\n",
        "input_shape = (img_rows, img_cols, 1)\n"
      ],
      "metadata": {
        "id": "JSJteGDnZ3lc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#separate x and y and convert to array\n",
        "\n",
        "#train dataset\n",
        "train_datasetX = np.array(train_dataset.iloc[:, 1:])\n",
        "train_datasetY = np.array(train_dataset.iloc[:, 0])\n",
        "\n",
        "#test dataset\n",
        "test_X = np.array(test_dataset.iloc[:, 1:])\n",
        "test_Y = np.array(test_dataset.iloc[:, 0])"
      ],
      "metadata": {
        "id": "oEC-RpzyZ-SF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#one hot encoding to the y variables of train and test data\n",
        "train_datasetY = to_categorical(train_datasetY)\n",
        "test_Y = to_categorical(test_Y)"
      ],
      "metadata": {
        "id": "hGEb-lDLaA1_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "batch_size = 64\n",
        "num_classes = test_Y.shape[1]\n"
      ],
      "metadata": {
        "id": "UlAx2pagaBPl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#split the train into train and validation (for later purposes)\n",
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "train_X, val_X, train_Y, val_Y = train_test_split(train_datasetX, train_datasetY, test_size=0.2, random_state=13)"
      ],
      "metadata": {
        "id": "3akYJqrHaCdW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(train_X.shape)\n",
        "print(val_X.shape)\n",
        "print(test_X.shape)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FEGfgK6XaEEL",
        "outputId": "55c7d76f-16f6-48e2-ff8e-2c56bec179c1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(48000, 784)\n",
            "(12000, 784)\n",
            "(10000, 784)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def prep_pixels(train_X, val_X, test_X, train_Y, val_Y, test_Y):\n",
        "  trainX = train_X.reshape(train_X.shape[0], img_rows, img_cols, 1)\n",
        "  valX = val_X.reshape(val_X.shape[0], img_rows, img_cols, 1)\n",
        "  testX = test_X.reshape(test_X.shape[0], img_rows, img_cols, 1)\n",
        "  trainY = train_Y\n",
        "  valY = val_Y\n",
        "  testY = test_Y\n",
        "\t# convert from integers to floats\n",
        "  trainX = trainX.astype('float32')\n",
        "  valX = valX.astype('float32')\n",
        "  testX = testX.astype('float32')\n",
        "\t# normalize to range 0-1\n",
        "  trainX = trainX / 255.0\n",
        "  valX = valX / 255.0\n",
        "  testX = testX / 255.0\n",
        "\n",
        "\t# return normalized images\n",
        "  return trainX, valX, testX, trainY, valY, testY"
      ],
      "metadata": {
        "id": "RI0R7edGaGIb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **We have divided the train, test, and validation for this particular section. There are 10,000 photos for testing and 60,000 images for training in the dataset. Following the splitting, a validation set comprising 12,000 rows is produced. After that, the sets are formatted to make them easier to read. It measures 28 by 28.**"
      ],
      "metadata": {
        "id": "bPn_KVMqajMO"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def summarize_diagnostics(history):\n",
        "\t# plot loss\n",
        "\tpyplot.figure(figsize=(16,10))\n",
        "\tpyplot.subplot(211)\n",
        "\tpyplot.title('Cross Entropy Loss')\n",
        "\tpyplot.plot(history.history['loss'], color='blue', label='train')\n",
        "\tpyplot.plot(history.history['val_loss'], color='orange', label='test')\n",
        "\t# plot accuracy\n",
        "\tpyplot.subplot(212)\n",
        "\tpyplot.title('Classification Accuracy')\n",
        "\tpyplot.plot(history.history['accuracy'], color='blue', label='train')\n",
        "\tpyplot.plot(history.history['val_accuracy'], color='orange', label='test')\n"
      ],
      "metadata": {
        "id": "fzywgmn6aIHT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **This function plots the loss and accuracy later on after the training is done.**"
      ],
      "metadata": {
        "id": "RBksDYEKavHr"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Performing and Saving Augmentation, Utilizing Test Harness**"
      ],
      "metadata": {
        "id": "ptgRPUVUa7aC"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "shift_fraction = 0.005"
      ],
      "metadata": {
        "id": "8eDChaA4axm4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def run_test_harness(model_x, epochs):\n",
        "  trainX, valX, testX, trainY, valY, testY = prep_pixels(train_X, val_X, test_X, train_Y, val_Y, test_Y)\n",
        "  model = model_x\n",
        "  model.summary()\n",
        "\n",
        "  #Image Augmentation\n",
        "  datagen = ImageDataGenerator(width_shift_range=shift_fraction,height_shift_range=shift_fraction,horizontal_flip=True)\n",
        "\n",
        "  it_train = datagen.flow(trainX, trainY, batch_size=batch_size)\n",
        "\t# prepare iterator\n",
        "  it_val = datagen.flow(valX, valY, batch_size=batch_size)\n",
        "  # fit model\n",
        "  steps = int(trainX.shape[0] / batch_size)\n",
        "  history = model.fit_generator(it_train, steps_per_epoch=steps, epochs=epochs, validation_data=it_val, verbose=1)\n",
        "  # evaluate model\n",
        "  _, acc = model.evaluate(testX, testY, verbose=1)\n",
        "  print('Accuracy:')\n",
        "  print('> %.3f' % (acc * 100.0))\n",
        "  # learning curves\n",
        "  summarize_diagnostics(history)"
      ],
      "metadata": {
        "id": "tXUPRqvBa_Ux"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**For this part the train, test, and validation data are feed into data generator, which makes it easier for the sets to be preprocessed and then trained later on.**\n",
        "\n",
        "**The ImageDataGenerator() function is used where the augmentation is done. Featurewise Standardization, ZCA Whitening, Shift Range, and Flips where used.**\n",
        "\n",
        "**Afterwards a batch of augmented images were saved in local Google drive.**\n",
        "\n",
        "**The run_test_harness() contains the entire functions needed for training where it only needs to take the model name and it will automatically call the other tasks needed for this activity.**"
      ],
      "metadata": {
        "id": "iq8umb32bNTK"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **THIRD BASELINE MODEL**"
      ],
      "metadata": {
        "id": "avQWmISZber7"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras.callbacks import EarlyStopping\n"
      ],
      "metadata": {
        "id": "aY50IojabAjQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# define cnn model\n",
        "def define_model_3():\n",
        "\tmodel3 = Sequential()\n",
        "\tmodel3.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))\n",
        "\tmodel3.add(BatchNormalization())\n",
        "\tmodel3.add(Conv2D(32, (3, 3), activation='relu', padding='same'))\n",
        "\tmodel3.add(BatchNormalization())\n",
        "\tmodel3.add(MaxPooling2D((2, 2)))\n",
        "\tmodel3.add(Dropout(0.25))\n",
        "\n",
        "\tmodel3.add(Conv2D(64, (3, 3), activation='relu', padding='same'))\n",
        "\tmodel3.add(BatchNormalization())\n",
        "\tmodel3.add(Conv2D(64, (3, 3), activation='relu', padding='same'))\n",
        "\tmodel3.add(BatchNormalization())\n",
        "\tmodel3.add(MaxPooling2D((2, 2)))\n",
        "\tmodel3.add(Dropout(0.25))\n",
        "\n",
        "\tmodel3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))\n",
        "\tmodel3.add(BatchNormalization())\n",
        "\tmodel3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))\n",
        "\tmodel3.add(BatchNormalization())\n",
        "\tmodel3.add(MaxPooling2D((2, 2)))\n",
        "\tmodel3.add(Dropout(0.25))\n",
        "\n",
        "\tmodel3.add(Conv2D(256, (3, 3), activation='relu', padding='same'))\n",
        "\tmodel3.add(BatchNormalization())\n",
        "\tmodel3.add(Conv2D(256, (3, 3), activation='relu', padding='same'))\n",
        "\tmodel3.add(BatchNormalization())\n",
        "\tmodel3.add(MaxPooling2D((2, 2)))\n",
        "\tmodel3.add(Dropout(0.25))\n",
        "\n",
        "\n",
        "\n",
        "\tmodel3.add(Flatten())\n",
        "\tmodel3.add(Dropout(0.25))\n",
        "\tmodel3.add(Dense(512, activation='relu'))\n",
        "\tmodel3.add(Dropout(0.25))\n",
        "\tmodel3.add(Dense(128, activation='relu'))\n",
        "\tmodel3.add(Dropout(0.25))\n",
        "\tmodel3.add(Dense(10, activation='softmax'))\n",
        "\n",
        "\t# compile model\n",
        "\n",
        "\tmodel3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n",
        "\treturn model3"
      ],
      "metadata": {
        "id": "C2Pms7RTbk6-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "early_stop = EarlyStopping(monitor='val_loss', patience=2)"
      ],
      "metadata": {
        "id": "dFgyzPg6bnA-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model3= define_model_3()"
      ],
      "metadata": {
        "id": "xNEtR_Cybo48"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "run_test_harness(model3, 100)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "mhp-Mg9tbrW6",
        "outputId": "37f8216d-dc00-4187-b11c-59b176d87603"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"sequential\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " conv2d (Conv2D)             (None, 28, 28, 32)        320       \n",
            "                                                                 \n",
            " batch_normalization (Batch  (None, 28, 28, 32)        128       \n",
            " Normalization)                                                  \n",
            "                                                                 \n",
            " conv2d_1 (Conv2D)           (None, 28, 28, 32)        9248      \n",
            "                                                                 \n",
            " batch_normalization_1 (Bat  (None, 28, 28, 32)        128       \n",
            " chNormalization)                                                \n",
            "                                                                 \n",
            " max_pooling2d (MaxPooling2  (None, 14, 14, 32)        0         \n",
            " D)                                                              \n",
            "                                                                 \n",
            " dropout (Dropout)           (None, 14, 14, 32)        0         \n",
            "                                                                 \n",
            " conv2d_2 (Conv2D)           (None, 14, 14, 64)        18496     \n",
            "                                                                 \n",
            " batch_normalization_2 (Bat  (None, 14, 14, 64)        256       \n",
            " chNormalization)                                                \n",
            "                                                                 \n",
            " conv2d_3 (Conv2D)           (None, 14, 14, 64)        36928     \n",
            "                                                                 \n",
            " batch_normalization_3 (Bat  (None, 14, 14, 64)        256       \n",
            " chNormalization)                                                \n",
            "                                                                 \n",
            " max_pooling2d_1 (MaxPoolin  (None, 7, 7, 64)          0         \n",
            " g2D)                                                            \n",
            "                                                                 \n",
            " dropout_1 (Dropout)         (None, 7, 7, 64)          0         \n",
            "                                                                 \n",
            " conv2d_4 (Conv2D)           (None, 7, 7, 128)         73856     \n",
            "                                                                 \n",
            " batch_normalization_4 (Bat  (None, 7, 7, 128)         512       \n",
            " chNormalization)                                                \n",
            "                                                                 \n",
            " conv2d_5 (Conv2D)           (None, 7, 7, 128)         147584    \n",
            "                                                                 \n",
            " batch_normalization_5 (Bat  (None, 7, 7, 128)         512       \n",
            " chNormalization)                                                \n",
            "                                                                 \n",
            " max_pooling2d_2 (MaxPoolin  (None, 3, 3, 128)         0         \n",
            " g2D)                                                            \n",
            "                                                                 \n",
            " dropout_2 (Dropout)         (None, 3, 3, 128)         0         \n",
            "                                                                 \n",
            " conv2d_6 (Conv2D)           (None, 3, 3, 256)         295168    \n",
            "                                                                 \n",
            " batch_normalization_6 (Bat  (None, 3, 3, 256)         1024      \n",
            " chNormalization)                                                \n",
            "                                                                 \n",
            " conv2d_7 (Conv2D)           (None, 3, 3, 256)         590080    \n",
            "                                                                 \n",
            " batch_normalization_7 (Bat  (None, 3, 3, 256)         1024      \n",
            " chNormalization)                                                \n",
            "                                                                 \n",
            " max_pooling2d_3 (MaxPoolin  (None, 1, 1, 256)         0         \n",
            " g2D)                                                            \n",
            "                                                                 \n",
            " dropout_3 (Dropout)         (None, 1, 1, 256)         0         \n",
            "                                                                 \n",
            " flatten (Flatten)           (None, 256)               0         \n",
            "                                                                 \n",
            " dropout_4 (Dropout)         (None, 256)               0         \n",
            "                                                                 \n",
            " dense (Dense)               (None, 512)               131584    \n",
            "                                                                 \n",
            " dropout_5 (Dropout)         (None, 512)               0         \n",
            "                                                                 \n",
            " dense_1 (Dense)             (None, 128)               65664     \n",
            "                                                                 \n",
            " dropout_6 (Dropout)         (None, 128)               0         \n",
            "                                                                 \n",
            " dense_2 (Dense)             (None, 10)                1290      \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 1374058 (5.24 MB)\n",
            "Trainable params: 1372138 (5.23 MB)\n",
            "Non-trainable params: 1920 (7.50 KB)\n",
            "_________________________________________________________________\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-19-cbaefd22cec5>:14: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.\n",
            "  history = model.fit_generator(it_train, steps_per_epoch=steps, epochs=epochs, validation_data=it_val, verbose=1)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/100\n",
            "750/750 [==============================] - 42s 35ms/step - loss: 0.6801 - accuracy: 0.7539 - val_loss: 0.4355 - val_accuracy: 0.8425\n",
            "Epoch 2/100\n",
            "750/750 [==============================] - 28s 37ms/step - loss: 0.4042 - accuracy: 0.8571 - val_loss: 0.3408 - val_accuracy: 0.8708\n",
            "Epoch 3/100\n",
            "750/750 [==============================] - 26s 35ms/step - loss: 0.3362 - accuracy: 0.8815 - val_loss: 0.2830 - val_accuracy: 0.8957\n",
            "Epoch 4/100\n",
            "750/750 [==============================] - 24s 32ms/step - loss: 0.3102 - accuracy: 0.8935 - val_loss: 0.3227 - val_accuracy: 0.8844\n",
            "Epoch 5/100\n",
            "750/750 [==============================] - 26s 34ms/step - loss: 0.2897 - accuracy: 0.8987 - val_loss: 0.2483 - val_accuracy: 0.9120\n",
            "Epoch 6/100\n",
            "750/750 [==============================] - 27s 35ms/step - loss: 0.2697 - accuracy: 0.9054 - val_loss: 0.3058 - val_accuracy: 0.8939\n",
            "Epoch 7/100\n",
            "750/750 [==============================] - 25s 33ms/step - loss: 0.2588 - accuracy: 0.9097 - val_loss: 0.2339 - val_accuracy: 0.9147\n",
            "Epoch 8/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.2449 - accuracy: 0.9150 - val_loss: 0.2545 - val_accuracy: 0.9054\n",
            "Epoch 9/100\n",
            "750/750 [==============================] - 25s 33ms/step - loss: 0.2370 - accuracy: 0.9168 - val_loss: 0.2322 - val_accuracy: 0.9207\n",
            "Epoch 10/100\n",
            "750/750 [==============================] - 25s 34ms/step - loss: 0.2266 - accuracy: 0.9196 - val_loss: 0.2513 - val_accuracy: 0.9173\n",
            "Epoch 11/100\n",
            "750/750 [==============================] - 32s 42ms/step - loss: 0.2203 - accuracy: 0.9232 - val_loss: 0.2229 - val_accuracy: 0.9250\n",
            "Epoch 12/100\n",
            "750/750 [==============================] - 24s 33ms/step - loss: 0.2107 - accuracy: 0.9263 - val_loss: 0.2191 - val_accuracy: 0.9258\n",
            "Epoch 13/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.2056 - accuracy: 0.9271 - val_loss: 0.2253 - val_accuracy: 0.9238\n",
            "Epoch 14/100\n",
            "750/750 [==============================] - 24s 33ms/step - loss: 0.1982 - accuracy: 0.9307 - val_loss: 0.2159 - val_accuracy: 0.9255\n",
            "Epoch 15/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.1897 - accuracy: 0.9330 - val_loss: 0.2068 - val_accuracy: 0.9243\n",
            "Epoch 16/100\n",
            "750/750 [==============================] - 26s 35ms/step - loss: 0.1863 - accuracy: 0.9341 - val_loss: 0.2067 - val_accuracy: 0.9292\n",
            "Epoch 17/100\n",
            "750/750 [==============================] - 25s 33ms/step - loss: 0.1826 - accuracy: 0.9349 - val_loss: 0.2068 - val_accuracy: 0.9264\n",
            "Epoch 18/100\n",
            "750/750 [==============================] - 27s 35ms/step - loss: 0.1742 - accuracy: 0.9369 - val_loss: 0.1977 - val_accuracy: 0.9296\n",
            "Epoch 19/100\n",
            "750/750 [==============================] - 26s 35ms/step - loss: 0.1713 - accuracy: 0.9391 - val_loss: 0.1861 - val_accuracy: 0.9362\n",
            "Epoch 20/100\n",
            "750/750 [==============================] - 25s 34ms/step - loss: 0.1653 - accuracy: 0.9405 - val_loss: 0.1889 - val_accuracy: 0.9335\n",
            "Epoch 21/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.1615 - accuracy: 0.9442 - val_loss: 0.1867 - val_accuracy: 0.9333\n",
            "Epoch 22/100\n",
            "750/750 [==============================] - 24s 32ms/step - loss: 0.1556 - accuracy: 0.9455 - val_loss: 0.1924 - val_accuracy: 0.9343\n",
            "Epoch 23/100\n",
            "750/750 [==============================] - 26s 34ms/step - loss: 0.1563 - accuracy: 0.9446 - val_loss: 0.1951 - val_accuracy: 0.9348\n",
            "Epoch 24/100\n",
            "750/750 [==============================] - 26s 35ms/step - loss: 0.1494 - accuracy: 0.9471 - val_loss: 0.1949 - val_accuracy: 0.9317\n",
            "Epoch 25/100\n",
            "750/750 [==============================] - 24s 32ms/step - loss: 0.1460 - accuracy: 0.9467 - val_loss: 0.1918 - val_accuracy: 0.9323\n",
            "Epoch 26/100\n",
            "750/750 [==============================] - 25s 33ms/step - loss: 0.1431 - accuracy: 0.9476 - val_loss: 0.2139 - val_accuracy: 0.9309\n",
            "Epoch 27/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.1369 - accuracy: 0.9490 - val_loss: 0.1910 - val_accuracy: 0.9342\n",
            "Epoch 28/100\n",
            "750/750 [==============================] - 25s 33ms/step - loss: 0.1402 - accuracy: 0.9503 - val_loss: 0.2034 - val_accuracy: 0.9322\n",
            "Epoch 29/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.1351 - accuracy: 0.9529 - val_loss: 0.1837 - val_accuracy: 0.9388\n",
            "Epoch 30/100\n",
            "750/750 [==============================] - 25s 33ms/step - loss: 0.1317 - accuracy: 0.9529 - val_loss: 0.1845 - val_accuracy: 0.9375\n",
            "Epoch 31/100\n",
            "750/750 [==============================] - 26s 35ms/step - loss: 0.1261 - accuracy: 0.9556 - val_loss: 0.1969 - val_accuracy: 0.9360\n",
            "Epoch 32/100\n",
            "750/750 [==============================] - 26s 35ms/step - loss: 0.1269 - accuracy: 0.9545 - val_loss: 0.1955 - val_accuracy: 0.9355\n",
            "Epoch 33/100\n",
            "750/750 [==============================] - 24s 32ms/step - loss: 0.1216 - accuracy: 0.9556 - val_loss: 0.1920 - val_accuracy: 0.9366\n",
            "Epoch 34/100\n",
            "750/750 [==============================] - 27s 35ms/step - loss: 0.1174 - accuracy: 0.9567 - val_loss: 0.1983 - val_accuracy: 0.9398\n",
            "Epoch 35/100\n",
            "750/750 [==============================] - 24s 33ms/step - loss: 0.1178 - accuracy: 0.9579 - val_loss: 0.2087 - val_accuracy: 0.9352\n",
            "Epoch 36/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.1156 - accuracy: 0.9586 - val_loss: 0.2130 - val_accuracy: 0.9336\n",
            "Epoch 37/100\n",
            "750/750 [==============================] - 24s 32ms/step - loss: 0.1181 - accuracy: 0.9588 - val_loss: 0.1923 - val_accuracy: 0.9384\n",
            "Epoch 38/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.1101 - accuracy: 0.9614 - val_loss: 0.1950 - val_accuracy: 0.9367\n",
            "Epoch 39/100\n",
            "750/750 [==============================] - 26s 35ms/step - loss: 0.1076 - accuracy: 0.9609 - val_loss: 0.2025 - val_accuracy: 0.9355\n",
            "Epoch 40/100\n",
            "750/750 [==============================] - 25s 34ms/step - loss: 0.1050 - accuracy: 0.9624 - val_loss: 0.2008 - val_accuracy: 0.9360\n",
            "Epoch 41/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.1050 - accuracy: 0.9614 - val_loss: 0.2090 - val_accuracy: 0.9362\n",
            "Epoch 42/100\n",
            "750/750 [==============================] - 24s 32ms/step - loss: 0.1027 - accuracy: 0.9628 - val_loss: 0.2004 - val_accuracy: 0.9377\n",
            "Epoch 43/100\n",
            "750/750 [==============================] - 25s 34ms/step - loss: 0.1048 - accuracy: 0.9623 - val_loss: 0.2060 - val_accuracy: 0.9357\n",
            "Epoch 44/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.0975 - accuracy: 0.9661 - val_loss: 0.2081 - val_accuracy: 0.9336\n",
            "Epoch 45/100\n",
            "750/750 [==============================] - 24s 32ms/step - loss: 0.0962 - accuracy: 0.9657 - val_loss: 0.1894 - val_accuracy: 0.9398\n",
            "Epoch 46/100\n",
            "750/750 [==============================] - 25s 33ms/step - loss: 0.0986 - accuracy: 0.9652 - val_loss: 0.1969 - val_accuracy: 0.9407\n",
            "Epoch 47/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.0947 - accuracy: 0.9657 - val_loss: 0.1984 - val_accuracy: 0.9392\n",
            "Epoch 48/100\n",
            "750/750 [==============================] - 26s 35ms/step - loss: 0.0935 - accuracy: 0.9661 - val_loss: 0.1959 - val_accuracy: 0.9352\n",
            "Epoch 49/100\n",
            "750/750 [==============================] - 24s 33ms/step - loss: 0.0930 - accuracy: 0.9661 - val_loss: 0.2081 - val_accuracy: 0.9399\n",
            "Epoch 50/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.0890 - accuracy: 0.9679 - val_loss: 0.2065 - val_accuracy: 0.9401\n",
            "Epoch 51/100\n",
            "750/750 [==============================] - 24s 33ms/step - loss: 0.0852 - accuracy: 0.9691 - val_loss: 0.1954 - val_accuracy: 0.9385\n",
            "Epoch 52/100\n",
            "750/750 [==============================] - 26s 34ms/step - loss: 0.0879 - accuracy: 0.9693 - val_loss: 0.2122 - val_accuracy: 0.9383\n",
            "Epoch 53/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.0829 - accuracy: 0.9695 - val_loss: 0.2061 - val_accuracy: 0.9401\n",
            "Epoch 54/100\n",
            "750/750 [==============================] - 25s 33ms/step - loss: 0.0834 - accuracy: 0.9703 - val_loss: 0.2065 - val_accuracy: 0.9398\n",
            "Epoch 55/100\n",
            "750/750 [==============================] - 26s 34ms/step - loss: 0.0894 - accuracy: 0.9687 - val_loss: 0.2054 - val_accuracy: 0.9376\n",
            "Epoch 56/100\n",
            "750/750 [==============================] - 27s 35ms/step - loss: 0.0809 - accuracy: 0.9703 - val_loss: 0.2221 - val_accuracy: 0.9417\n",
            "Epoch 57/100\n",
            "750/750 [==============================] - 25s 33ms/step - loss: 0.0807 - accuracy: 0.9709 - val_loss: 0.2129 - val_accuracy: 0.9395\n",
            "Epoch 58/100\n",
            "750/750 [==============================] - 26s 35ms/step - loss: 0.0825 - accuracy: 0.9714 - val_loss: 0.2293 - val_accuracy: 0.9403\n",
            "Epoch 59/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.0752 - accuracy: 0.9725 - val_loss: 0.2248 - val_accuracy: 0.9365\n",
            "Epoch 60/100\n",
            "750/750 [==============================] - 25s 33ms/step - loss: 0.0811 - accuracy: 0.9719 - val_loss: 0.2258 - val_accuracy: 0.9361\n",
            "Epoch 61/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.0817 - accuracy: 0.9706 - val_loss: 0.2058 - val_accuracy: 0.9412\n",
            "Epoch 62/100\n",
            "750/750 [==============================] - 27s 35ms/step - loss: 0.0746 - accuracy: 0.9732 - val_loss: 0.2128 - val_accuracy: 0.9388\n",
            "Epoch 63/100\n",
            "750/750 [==============================] - 25s 33ms/step - loss: 0.0763 - accuracy: 0.9731 - val_loss: 0.2174 - val_accuracy: 0.9413\n",
            "Epoch 64/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.0786 - accuracy: 0.9722 - val_loss: 0.2184 - val_accuracy: 0.9399\n",
            "Epoch 65/100\n",
            "750/750 [==============================] - 26s 35ms/step - loss: 0.0717 - accuracy: 0.9745 - val_loss: 0.2192 - val_accuracy: 0.9399\n",
            "Epoch 66/100\n",
            "750/750 [==============================] - 26s 34ms/step - loss: 0.0725 - accuracy: 0.9739 - val_loss: 0.2376 - val_accuracy: 0.9385\n",
            "Epoch 67/100\n",
            "750/750 [==============================] - 27s 35ms/step - loss: 0.0746 - accuracy: 0.9744 - val_loss: 0.2125 - val_accuracy: 0.9406\n",
            "Epoch 68/100\n",
            "750/750 [==============================] - 25s 33ms/step - loss: 0.0708 - accuracy: 0.9746 - val_loss: 0.2194 - val_accuracy: 0.9416\n",
            "Epoch 69/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.0755 - accuracy: 0.9745 - val_loss: 0.2230 - val_accuracy: 0.9396\n",
            "Epoch 70/100\n",
            "750/750 [==============================] - 26s 35ms/step - loss: 0.0699 - accuracy: 0.9753 - val_loss: 0.2234 - val_accuracy: 0.9419\n",
            "Epoch 71/100\n",
            "750/750 [==============================] - 26s 34ms/step - loss: 0.0703 - accuracy: 0.9754 - val_loss: 0.2060 - val_accuracy: 0.9413\n",
            "Epoch 72/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.0636 - accuracy: 0.9773 - val_loss: 0.2456 - val_accuracy: 0.9401\n",
            "Epoch 73/100\n",
            "750/750 [==============================] - 25s 33ms/step - loss: 0.0674 - accuracy: 0.9763 - val_loss: 0.2243 - val_accuracy: 0.9401\n",
            "Epoch 74/100\n",
            "750/750 [==============================] - 25s 34ms/step - loss: 0.0713 - accuracy: 0.9756 - val_loss: 0.2213 - val_accuracy: 0.9388\n",
            "Epoch 75/100\n",
            "750/750 [==============================] - 27s 35ms/step - loss: 0.0658 - accuracy: 0.9764 - val_loss: 0.2115 - val_accuracy: 0.9387\n",
            "Epoch 76/100\n",
            "750/750 [==============================] - 24s 32ms/step - loss: 0.0641 - accuracy: 0.9769 - val_loss: 0.2223 - val_accuracy: 0.9413\n",
            "Epoch 77/100\n",
            "750/750 [==============================] - 26s 35ms/step - loss: 0.0603 - accuracy: 0.9787 - val_loss: 0.2491 - val_accuracy: 0.9419\n",
            "Epoch 78/100\n",
            "750/750 [==============================] - 26s 35ms/step - loss: 0.0620 - accuracy: 0.9775 - val_loss: 0.2219 - val_accuracy: 0.9406\n",
            "Epoch 79/100\n",
            "750/750 [==============================] - 25s 33ms/step - loss: 0.0603 - accuracy: 0.9783 - val_loss: 0.2523 - val_accuracy: 0.9376\n",
            "Epoch 80/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.0613 - accuracy: 0.9786 - val_loss: 0.2300 - val_accuracy: 0.9408\n",
            "Epoch 81/100\n",
            "750/750 [==============================] - 26s 34ms/step - loss: 0.0596 - accuracy: 0.9791 - val_loss: 0.2470 - val_accuracy: 0.9388\n",
            "Epoch 82/100\n",
            "750/750 [==============================] - 24s 32ms/step - loss: 0.0599 - accuracy: 0.9787 - val_loss: 0.2344 - val_accuracy: 0.9397\n",
            "Epoch 83/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.0594 - accuracy: 0.9786 - val_loss: 0.2536 - val_accuracy: 0.9385\n",
            "Epoch 84/100\n",
            "750/750 [==============================] - 24s 32ms/step - loss: 0.0605 - accuracy: 0.9796 - val_loss: 0.2297 - val_accuracy: 0.9397\n",
            "Epoch 85/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.0659 - accuracy: 0.9775 - val_loss: 0.2363 - val_accuracy: 0.9423\n",
            "Epoch 86/100\n",
            "750/750 [==============================] - 27s 35ms/step - loss: 0.0585 - accuracy: 0.9789 - val_loss: 0.2371 - val_accuracy: 0.9393\n",
            "Epoch 87/100\n",
            "750/750 [==============================] - 25s 33ms/step - loss: 0.0544 - accuracy: 0.9806 - val_loss: 0.2567 - val_accuracy: 0.9403\n",
            "Epoch 88/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.0550 - accuracy: 0.9810 - val_loss: 0.2277 - val_accuracy: 0.9392\n",
            "Epoch 89/100\n",
            "750/750 [==============================] - 26s 35ms/step - loss: 0.0590 - accuracy: 0.9795 - val_loss: 0.2309 - val_accuracy: 0.9388\n",
            "Epoch 90/100\n",
            "750/750 [==============================] - 25s 33ms/step - loss: 0.0565 - accuracy: 0.9795 - val_loss: 0.2264 - val_accuracy: 0.9409\n",
            "Epoch 91/100\n",
            "750/750 [==============================] - 27s 35ms/step - loss: 0.0531 - accuracy: 0.9807 - val_loss: 0.2528 - val_accuracy: 0.9405\n",
            "Epoch 92/100\n",
            "750/750 [==============================] - 26s 35ms/step - loss: 0.0550 - accuracy: 0.9805 - val_loss: 0.2514 - val_accuracy: 0.9402\n",
            "Epoch 93/100\n",
            "750/750 [==============================] - 25s 33ms/step - loss: 0.0556 - accuracy: 0.9804 - val_loss: 0.2413 - val_accuracy: 0.9406\n",
            "Epoch 94/100\n",
            "750/750 [==============================] - 26s 35ms/step - loss: 0.0532 - accuracy: 0.9811 - val_loss: 0.2385 - val_accuracy: 0.9410\n",
            "Epoch 95/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.0523 - accuracy: 0.9826 - val_loss: 0.2223 - val_accuracy: 0.9427\n",
            "Epoch 96/100\n",
            "750/750 [==============================] - 24s 32ms/step - loss: 0.0515 - accuracy: 0.9824 - val_loss: 0.2263 - val_accuracy: 0.9400\n",
            "Epoch 97/100\n",
            "750/750 [==============================] - 27s 35ms/step - loss: 0.0508 - accuracy: 0.9820 - val_loss: 0.2291 - val_accuracy: 0.9409\n",
            "Epoch 98/100\n",
            "750/750 [==============================] - 24s 33ms/step - loss: 0.0525 - accuracy: 0.9815 - val_loss: 0.2425 - val_accuracy: 0.9402\n",
            "Epoch 99/100\n",
            "750/750 [==============================] - 27s 36ms/step - loss: 0.0543 - accuracy: 0.9815 - val_loss: 0.2411 - val_accuracy: 0.9410\n",
            "Epoch 100/100\n",
            "750/750 [==============================] - 24s 32ms/step - loss: 0.0564 - accuracy: 0.9805 - val_loss: 0.2423 - val_accuracy: 0.9386\n",
            "313/313 [==============================] - 2s 7ms/step - loss: 0.2178 - accuracy: 0.9436\n",
            "Accuracy:\n",
            "> 94.360\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1600x1000 with 2 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAABRQAAANECAYAAADBuKMlAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAD7rElEQVR4nOzdeXhU5d3/8c9kT8gGhIQtEBbZNwVBQAEFRQUVq1YtCmL16fOIVsUuWn/ValWqtharVtS6tWpFccMNRRTcEARE2fcdEgiQhCRknfP748tkMmRhAkkmk7xf13Wuc+bMOWfuGTIk85n7vr8ux3EcAQAAAAAAAIAfQgLdAAAAAAAAAADBg0ARAAAAAAAAgN8IFAEAAAAAAAD4jUARAAAAAAAAgN8IFAEAAAAAAAD4jUARAAAAAAAAgN8IFAEAAAAAAAD4jUARAAAAAAAAgN8IFAEAAAAAAAD4jUARAAAAAAAAgN8IFAEAAE7C5s2b9atf/UqdO3dWVFSU4uPjNXz4cD3++OM6cuRIoJvnlz/96U9yuVxVLunp6TW+5muvvaYZM2bUfmPrUVpamsaPHx/oZgAAADQ4YYFuAAAAQLD68MMPdcUVVygyMlKTJk1Snz59VFRUpK+//lq//e1vtXr1aj377LOBbqbfnn76acXGxlbYn5iYWONrvfbaa1q1apVuu+22k28YAAAAGhQCRQAAgBOwdetWXXXVVerYsaM+//xztWnTpuy+qVOnatOmTfrwww+rPN/tdquoqEhRUVH10Vy/XH755UpKSqr3xy0oKFBERIRCQhg8AwAAEAz4qw0AAOAEPPLII8rNzdXzzz/vEyZ6dO3aVbfeemvZbZfLpZtvvlmvvvqqevfurcjISM2dO1eS9MMPP+iCCy5QfHy8YmNjNXr0aH333Xc+1ysuLtZ9992nU045RVFRUWrZsqXOPPNMzZs3r+yY9PR0TZkyRe3bt1dkZKTatGmjSy65RNu2bauV57xgwQK5XC698cYbevDBB9W+fXtFRUVp9OjR2rRpU9lxo0aN0ocffqjt27eXDZtOS0vzucbrr7+u//f//p/atWunmJgY5eTkSJLefPNNDRw4UNHR0UpKStI111yj3bt3+7TjuuuuU2xsrLZs2aKxY8eqWbNmatu2re6//345jiNJchxHaWlpuuSSSyo8j4KCAiUkJOhXv/rVSb8mJSUl+vOf/6wuXbooMjJSaWlp+sMf/qDCwkKf45YuXaqxY8cqKSlJ0dHR6tSpk66//nqfY15//XUNHDhQcXFxio+PV9++ffX444+fdBsBAABqGz0UAQAATsD777+vzp07a9iwYX6f8/nnn+uNN97QzTffrKSkJKWlpWn16tU666yzFB8fr9/97ncKDw/XM888o1GjRmnhwoUaMmSIJJvncPr06brhhhs0ePBg5eTkaOnSpVq+fLnOPfdcSdJll12m1atX65ZbblFaWpr27dunefPmaceOHWWBXnUOHjxYYV9YWFiFIc9/+ctfFBISot/85jfKzs7WI488ookTJ2rx4sWSpLvvvlvZ2dnatWuX/v73v0tShaHUf/7znxUREaHf/OY3KiwsVEREhF566SVNmTJFp59+uqZPn66MjAw9/vjj+uabb/TDDz/4tKO0tFTnn3++zjjjDD3yyCOaO3eu7r33XpWUlOj++++Xy+XSNddco0ceeUQHDx5UixYtys59//33lZOTo2uuuea4r8nx3HDDDXr55Zd1+eWX64477tDixYs1ffp0rV27Vu+8844kad++fTrvvPPUqlUr3XnnnUpMTNS2bdv09ttvl11n3rx5uvrqqzV69Gg9/PDDkqS1a9fqm2++8QmmAQAAGgQHAAAANZKdne1Ici655BK/z5HkhISEOKtXr/bZP2HCBCciIsLZvHlz2b49e/Y4cXFxzogRI8r29e/f3xk3blyV1z906JAjyXn00Uf9fyJH3XvvvY6kSpfu3buXHffFF184kpyePXs6hYWFZfsff/xxR5KzcuXKsn3jxo1zOnbsWOGxPNfo3Lmzk5+fX7a/qKjISU5Odvr06eMcOXKkbP8HH3zgSHLuueeesn2TJ092JDm33HJL2T632+2MGzfOiYiIcPbv3+84juOsX7/ekeQ8/fTTPm24+OKLnbS0NMftdlf7unTs2LHa13zFihWOJOeGG27w2f+b3/zGkeR8/vnnjuM4zjvvvONIcr7//vsqr3Xrrbc68fHxTklJSbVtAgAAaAgY8gwAAFBDnuG5cXFxNTpv5MiR6tWrV9nt0tJSffrpp5owYYI6d+5ctr9Nmzb6xS9+oa+//rrssRITE7V69Wpt3Lix0mtHR0crIiJCCxYs0KFDh2r6lCRJb731lubNm+ezvPjiixWOmzJliiIiIspun3XWWZKkLVu2+P1YkydPVnR0dNntpUuXat++fbrpppt85pUcN26cevToUel8lDfffHPZtmdIeVFRkT777DNJUrdu3TRkyBC9+uqrZccdPHhQH3/8sSZOnCiXy+V3eyvz0UcfSZKmTZvms/+OO+6QpLI2e3pWfvDBByouLq70WomJicrLy/MZwg4AANBQESgCAADUUHx8vCTp8OHDNTqvU6dOPrf379+v/Px8de/evcKxPXv2lNvt1s6dOyVJ999/v7KystStWzf17dtXv/3tb/XTTz+VHR8ZGamHH35YH3/8sVJSUjRixAg98sgjSk9P97t9I0aM0JgxY3yWoUOHVjiuQ4cOPrebN28uSTUKMo99LbZv3y5Jlb4WPXr0KLvfIyQkxCeElSxAlOQzZ+SkSZP0zTfflJ3/5ptvqri4WNdee63fba3K9u3bFRISoq5du/rsb926tRITE8sec+TIkbrssst03333KSkpSZdccolefPFFn3kWb7rpJnXr1k0XXHCB2rdvr+uvv75sjk0AAICGhkARAACghuLj49W2bVutWrWqRueV75FXUyNGjNDmzZv1wgsvqE+fPvrXv/6l0047Tf/617/Kjrntttu0YcMGTZ8+XVFRUfrjH/+onj176ocffjjhx61MaGhopfudowVR/HEyr0VNXHXVVQoPDy/rpfjKK69o0KBBlQaXJ+p4PR1dLpdmz56tRYsW6eabb9bu3bt1/fXXa+DAgcrNzZUkJScna8WKFZozZ44uvvhiffHFF7rgggs0efLkWmsnAABAbSFQBAAAOAHjx4/X5s2btWjRohO+RqtWrRQTE6P169dXuG/dunUKCQlRampq2b4WLVpoypQp+u9//6udO3eqX79++tOf/uRzXpcuXXTHHXfo008/1apVq1RUVKS//e1vJ9zGE1XT4cQdO3aUpEpfi/Xr15fd7+F2uysMsd6wYYMk+RSgadGihcaNG6dXX31V27dv1zfffFMrvRM9bXa73RWGoWdkZCgrK6tCm8844ww9+OCDWrp0qV599VWtXr1ar7/+etn9ERERuuiii/TPf/5Tmzdv1q9+9Sv9+9//9qmgDQAA0BAQKAIAAJyA3/3ud2rWrJluuOEGZWRkVLh/8+bNevzxx6u9RmhoqM477zy99957PsN0MzIy9Nprr+nMM88sG1594MABn3NjY2PVtWvXsmGz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IEBLly517bMsS8XFxQoMDJS/v3+Dt2PBggWaOHGivv766yrViCUlJZKkoKCgBn/c2rr88sv1448/aseOHVq8eLHGjBnjtbYAAAC0ZAHebgAAAEBT2759uy699FJ16tRJX331ldq1a+e6b+bMmdq6das+/vhjL7bQU2BgYJV9Bw4cUJ8+fTz22Ww2hYSENFWzPHgzSJSkgoICffDBB5ozZ45effVVzZs3r9kGigUFBQoPD/d2MwAAAOqNLs8AAKDVefTRR5Wfn6+XX37ZI0x06t69u2655ZYaz8/OztZf/vIX9e/fXxEREYqKitK4ceP022+/VTn26aefVt++fRUWFqY2bdpoyJAheuutt1z3Hz58WLfeeqs6d+6s4OBgJSQk6A9/+IN+/vln1zEVx1D85ptvZLPZtH37dn388cey2Wyy2WzasWNHjWMobty4UZMmTVJ8fLxCQ0PVs2dP3XXXXa77d+7cqRtuuEE9e/ZUaGio2rZtq4kTJ3p0bX7ttdc0ceJESdLpp5/uetxvvvlGUvVjKB44cEDTp09XYmKiQkJCNHDgQL3++usexzjb/M9//lMvvviiunXrpuDgYJ188slatWpVjT+Dyt577z0dOXJEEydO1KWXXqpFixapqKioynFFRUW6//77dcIJJygkJETt2rXTRRddpG3btrmOcTgcevLJJ9W/f3+FhIQoPj5eZ599tn766SePNldXsWqz2XT//fe7bt9///2y2Wxav369LrvsMrVp00YjR46UJK1Zs0ZXXXWVq8t9UlKSrr76amVlZVW57p49ezR9+nQlJycrODhYXbp00fXXX6+SkhKlpaXJZrPpX//6V5XzfvzxR9lsNr399tu1fi0BAACOhQpFAADQ6nz00Ufq2rWrhg8fXq/z09LS9P7772vixInq0qWLMjIy9O9//1ujR4/W+vXrlZycLEl66aWXdPPNN+viiy/WLbfcoqKiIq1Zs0YrVqzQZZddJkm67rrrtGDBAt14443q06ePsrKytHTpUm3YsEEnnXRSlcfu3bu33njjDd12223q0KGD/vznP0uS4uPjlZmZWeX4NWvWaNSoUQoMDNSMGTPUuXNnbdu2TR999JEeeughSdKqVav0448/6tJLL1WHDh20Y8cOPf/88zrttNO0fv16hYWF6dRTT9XNN9+sp556Snfeead69+7tak91jhw5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UXp6usd9Fdtf3fN5+umnq1TchYeHS6oapFXnnHPO0f79+/XOO++49pWVlenpp59WRESERo8eXdenU6158+Zp1KhRuuSSS3TxxRd7LLfffrsk6e2335YkTZgwQQcPHtQzzzxT5TrO5z9hwgRZlqUHHnigxmOioqIUFxen7777zuP+5557rtbtdoaflV/3uXPnetz28/PTBRdcoI8++kg//fRTjW2SpICAAE2ePFnvvvuuXnvtNfXv318DBgyodZsAAABqiwpFAADQ6nTr1k1vvfWWLrnkEvXu3VtTpkxRv379VFJSoh9//FHz58/XVVddVeP55513nv7+979r2rRpGj58uNauXat58+a5KgKdzjrrLCUlJWnEiBFKTEzUhg0b9Mwzz+jcc89VZGSkcnJy1KFDB1188cUaOHCgIiIi9OWXX2rVqlUe1YLH46mnntLIkSN10kknacaMGerSpYt27Nihjz/+WL/++qvr+bzxxhuKjo5Wnz59tGzZMn355Zdq27atx7VOPPFE+fv765FHHlFubq6Cg4N1xhlnKCEhocrjzpgxQ//+97911VVXafXq1ercubMWLFigH374QXPnzj2uSXGcVqxYoa1bt+rGG2+s9v727dvrpJNO0rx58/S3v/1NU6ZM0X/+8x/NmjVLK1eu1KhRo1RQUKAvv/xSN9xwg84//3ydfvrpuvLKK/XUU09py5YtOvvss+VwOPT999/r9NNPdz3WNddco3/84x+65pprNGTIEH333XfavHlzrdseFRWlU089VY8++qhKS0vVvn17ffHFF9q+fXuVYx9++GF98cUXGj16tGbMmKHevXtr3759mj9/vpYuXaqYmBjXsVOmTNFTTz2lr7/+Wo888kjdXlAAAIBaIlAEAACt0h//+EetWbNGjz32mD744AM9//zzCg4O1oABA/T444/r2muvrfHcO++8UwUFBXrrrbf0zjvv6KSTTtLHH3+s2bNnexz3pz/9SfPmzdMTTzyh/Px8dejQQTfffLPuvvtuSVJYWJhuuOEGffHFF1q0aJEcDoe6d++u5557Ttdff32DPM+BAwdq+fLluueee/T888+rqKhInTp10qRJk1zHPPnkk/L399e8efNUVFSkESNG6Msvv9TYsWM9rpWUlKQXXnhBc+bM0fTp02W32/X1119XGyiGhobqm2++0ezZs/X6668r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cnK0evVq1zFfffWVHA6HUlNTm7zN9UWX52Zu1qxZmjp1qoYMGaKhQ4dq7ty5Kigo0LRp07zdNKBFmzlzpt566y198MEHioyMdI1lER0drdDQUEVHR2v69OmaNWuWYmNjFRUVpZtuuknDhg3TKaec4uXWAy1HZGSka+xSp/DwcLVt29a1n88i0DRuu+02DR8+XA8//LAmTZqklStX6sUXX9SLL74oSbLZbLr11lv1f//3f+rRo4e6dOmie+65R8nJybrgggu823ighRk/frweeughdezYUX379tUvv/yiJ554QldffbUkPo9AY8rPz9fWrVtdt7dv365ff/1VsbGx6tix4zE/e71799bZZ5+ta6+9Vi+88IJKS0t144036tJLL1VycrKXnlU9eHuaaRzb008/bXXs2NEKCgqyhg4dai1fvtzbTQJaPEnVLq+++qrrmCNHjlg33HCD1aZNGyssLMy68MILrX379nmv0UArMXr0aOuWW25x3eazCDSdjz76yOrXr58VHBxs9erVy3rxxRc97nc4HNY999xjJSYmWsHBwdaZZ55pbdq0yUutBVquvLw865ZbbrE6duxohYSEWF27drXuuusuq7i42HUMn0egcXz99dfVflecOnWqZVm1++xlZWVZkydPtiIiIqyoqChr2rRp1uHDh73wbOrPZlmW5aUsEwAAAAAAAICPYQxFAAAAAAAAALVGoAgAAAAAAACg1ggUAQAAAAAAANQagSIAAAAAAACAWiNQBAAAAAAAAFBrBIoAAAAAAAAAao1AEQAAAAAAAECtESgCAAAAAAAAqDUCRQAAAAAAAAC1RqAIAAAAAAAAoNYIFAEAAAAAAADUGoEiAAAAAAAAgFr7/1S0/qHUmcQSAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Finally, a considerably deeper network with three VGG blocks and 512 nodes in the final hidden layer is employed. Each had max norm per block, batch normalization per convolutional layer, and rising dropout rates. Adam was the optimizer, and 100 epochs were utilized. It outperformed the previous two models, as predicted, with 98% accuracy on the dataset and 94% validation. Although there is still some acceptable overfitting, it functioned nicely.**"
      ],
      "metadata": {
        "id": "kYxpRwp8zhvQ"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Finalization, saving of model, and testing on new images**"
      ],
      "metadata": {
        "id": "yZXsKzP0zoeG"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "trainX, valX, testX, trainY, valY, testY = prep_pixels(train_X, val_X, test_X, train_Y, val_Y, test_Y)"
      ],
      "metadata": {
        "id": "ZaP72qdnq4X_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model3.save('/content/drive/MyDrive/Models/saved_fashion.h5')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XFs-IlgVrTGW",
        "outputId": "cd4dbcda-fd3a-498c-faa5-2dd09e60d795"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3079: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.\n",
            "  saving_api.save_model(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model = load_model('/content/drive/MyDrive/Models/saved_fashion.h5')\n",
        "# evaluate model on test dataset\n",
        "_, acc = model.evaluate(testX, testY, verbose=0)\n",
        "print('> %.3f' % (acc * 100.0))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vvIXCBl6rcle",
        "outputId": "25d5e7fa-66eb-4b8e-a775-329d3508724b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "> 94.360\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# make a prediction for a new image.\n",
        "from tensorflow.keras.preprocessing.image import load_img\n",
        "from tensorflow.keras.preprocessing.image import img_to_array\n",
        "from keras.models import load_model\n",
        "\n",
        "# load and prepare the image\n",
        "def load_image(filename):\n",
        "  img = Image.open(filename).resize((224, 224))\n",
        "  plt.imshow(img)\n",
        "  plt.show()\n",
        "  img = load_img(filename, target_size=(28, 28))\n",
        "  img = img_to_array(img)\n",
        "  img = img[:,:,0]\n",
        "  img = img.reshape(1,28, 28, 1)\n",
        "  img = img.astype('float32')\n",
        "  img = img / 255.0\n",
        "  return img\n",
        "\n",
        "def run_example(filename):\n",
        "  img = load_image(filename)\n",
        "  model = load_model('/content/drive/MyDrive/Models/saved_fashion.h5')\n",
        "  result = np.argmax(model.predict(img), axis=1)\n",
        "  if result == 0:\n",
        "    print('Tshirt')\n",
        "  elif result == 1:\n",
        "    print('Top')\n",
        "  elif result == 2:\n",
        "    print('Pullover')\n",
        "  elif result == 3:\n",
        "    print('Dress')\n",
        "  elif result == 4:\n",
        "    print('Coat')\n",
        "  elif result == 5:\n",
        "    print('Sandal')\n",
        "  elif result == 6:\n",
        "    print('Shirt')\n",
        "  elif result == 7:\n",
        "    print('Snicker')\n",
        "  elif result == 8:\n",
        "    print('Bag')\n",
        "  else:\n",
        "    print('Ankle Boot')"
      ],
      "metadata": {
        "id": "HHoyzVlXrkTe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "run_example('/content/drive/MyDrive/Colab Notebooks/newimages/charlie.jpg')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 469
        },
        "id": "5av6hWfNsH6C",
        "outputId": "ac19cdbe-25d9-4034-9c32-3f4a785001c8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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fPnLX+b1r389hw4dQgjBZz7zmcZ6ay3vf//7OXjwIN1ul3vvvZfvfe97jW3Onz/PQw89xNTUFDMzM7ztbW9jbW3tOl7F7sDF+vKtb33rpmf0gQceaGzT9iU88sgj/PRP/zSTk5MsLCzwi7/4izz55JONbXbyPj/zzDO87nWvY2xsjIWFBX7913+doiiu56XsCHuOpP7gD/6A97znPfzGb/wGf/7nf85dd93F/fffz5kzZ25003Y9XvrSl3Ly5Mny88d//Mflune/+9384R/+IZ/61Kd4/PHHee6553jDG95wA1u7O7C+vs5dd93F7//+749c/9u//dv87u/+Lh/+8Id54oknGB8f5/7776ff75fbPPTQQ3znO9/h0Ucf5XOf+xxf/vKXecc73nG9LmHX4GJ9CfDAAw80ntFPfOITjfVtX8Ljjz/Oww8/zFe/+lUeffRR8jznvvvuY319vdzmYu+z1prXve51ZFnGV77yFT72sY/x0Y9+lPe///034pK2h91j+Jmf+Rn78MMPl7+11vbQoUP2kUceuYGt2v34jd/4DXvXXXeNXLe0tGTjOLaf+tSnymV/+Zd/aQF74sSJ69TC3Q/AfvrTny5/G2Ps4uKi/eAHP1guW1pasmma2k984hPWWmu/+93vWsD+2Z/9WbnN5z//eSuEsM8+++x1a/tuw3BfWmvtW97yFvsLv/ALW+7T9uVonDlzxgL28ccft9bu7H3+7//9v1sppT116lS5zYc+9CE7NTVlB4PB9b2Ai2BPSVJZlvH1r3+de++9t1wmpeTee+/lxIkTN7BlewPf+973OHToELfddhsPPfQQzzzzDABf//rXyfO80a933nknx44da/t1Gzz99NOcOnWq0W/T09PcfffdZb+dOHGCmZkZXv3qV5fb3HvvvUgpeeKJJ657m3c7HnvsMRYWFrjjjjt45zvfyblz58p1bV+OxvLyMgBzc3PAzt7nEydO8PKXv5wDBw6U29x///2srKzwne985zq2/uLYUyR19uxZtNaNjgU4cOAAp06dukGt2hu4++67+ehHP8oXvvAFPvShD/H000/z1/7aX2N1dZVTp06RJAkzMzONfdp+3R6hb7Z7Hk+dOsXCwkJjfRRFzM3NtX07hAceeID//J//M1/84hf5t//23/L444/z4IMPorUG2r4cBWMMv/Zrv8bP/dzP8bKXvQxgR+/zqVOnRj63Yd1uwp4s1dHi0vHggw+W31/xildw9913c8stt/Bf/+t/pdvt3sCWtWjh8KY3van8/vKXv5xXvOIVvOAFL+Cxxx7jta997Q1s2e7Fww8/zLe//e2GffknDXtKkpqfn0cptclL5fTp0ywuLt6gVu1NzMzM8KIXvYinnnqKxcVFsixjaWmpsU3br9sj9M12z+Pi4uImp56iKDh//nzbtxfBbbfdxvz8PE899RTQ9uUw3vWud/G5z32OP/qjP+LIkSPl8p28z4uLiyOf27BuN2FPkVSSJLzqVa/ii1/8YrnMGMMXv/hF7rnnnhvYsr2HtbU1vv/973Pw4EFe9apXEcdxo1+ffPJJnnnmmbZft8Hx48dZXFxs9NvKygpPPPFE2W/33HMPS0tLfP3rXy+3+dKXvoQxhrvvvvu6t3kv4cc//jHnzp3j4MGDQNuXAdZa3vWud/HpT3+aL33pSxw/fryxfifv8z333MNf/MVfNEj/0UcfZWpqipe85CXX50J2ihvtuXGp+OQnP2nTNLUf/ehH7Xe/+137jne8w87MzDS8VFpsxnvf+1772GOP2aefftr+yZ/8ib333nvt/Py8PXPmjLXW2n/wD/6BPXbsmP3Sl75kv/a1r9l77rnH3nPPPTe41Tceq6ur9hvf+Ib9xje+YQH7O7/zO/Yb3/iG/eEPf2ittfYDH/iAnZmZsZ/97Gftt771LfsLv/AL9vjx47bX65XHeOCBB+wrX/lK+8QTT9g//uM/trfffrt985vffKMu6YZhu75cXV21/+Sf/BN74sQJ+/TTT9v/+T//p/2pn/ope/vtt9t+v18eo+1La9/5znfa6elp+9hjj9mTJ0+Wn42NjXKbi73PRVHYl73sZfa+++6z3/zmN+0XvvAFu3//fvu+973vRlzStthzJGWttb/3e79njx07ZpMksT/zMz9jv/rVr97oJu16/PIv/7I9ePCgTZLEHj582P7yL/+yfeqpp8r1vV7P/sN/+A/t7OysHRsbs7/0S79kT548eQNbvDvwR3/0RxbY9HnLW95irXVu6P/yX/5Le+DAAZumqX3ta19rn3zyycYxzp07Z9/85jfbiYkJOzU1ZX/lV37Frq6u3oCrubHYri83NjbsfffdZ/fv32/jOLa33HKLffvb375p8tn2pR3Zh4D9yEc+Um6zk/f5Bz/4gX3wwQdtt9u18/Pz9r3vfa/N8/w6X83F0daTatGiRYsWuxZ7yibVokWLFi1uLrQk1aJFixYtdi1akmrRokWLFrsWLUm1aNGiRYtdi5akWrRo0aLFrkVLUi1atGjRYteiJakWLVq0aLFr0ZJUixYtWrTYtWhJqkWLFi1a7Fq0JNWiRYsWLXYtWpJq0aJFixa7Fv8/xLwb4t1B4vkAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 0s 461ms/step\n",
            "Bag\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **CONCLUSION:** <BR>\n",
        "### Doroteo:\n",
        "**This project demonstrated the value of CNN in picture categorization because it employs a distinct technique from densenet. Because of its practicality and doability, it is currently frequently employed in the AI business. In this exercise, a model that can identify the type of fashion item supplied as input is trained using the fashion MNIST. A good performance, with a validation accuracy of 94%, was shown in the results. To further increase the capabilities of the model, various techniques are used, including batch normalization, dropout, and picture augmentation.**\n"
      ],
      "metadata": {
        "id": "3dRVl7J7zx2m"
      }
    }
  ]
}
