#!/usr/bin/env python
# coding: utf-8

# In[18]:


import simpy
import random
import ipaddress
import scapy.all as scapy
import logging
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import joblib
from scapy.layers.inet import IP, TCP
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import yaml

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(levelname)s: %(message)s')

# Configuration for simulation
config = {
    'simulation': {
        'duration': 2000,  # Total duration of the simulation
        'normal_traffic_interval': 2,  # Interval between normal packets
        'attack_interval': 5,  # Interval between attack packets
    },
    'network': {
        'hmi_ip': '192.168.1.100',
        'plc_ip': '192.168.1.200',
        'modbus_port': 502,
    },
    'paths': {
        'dataset_output': 'balanced_modbus_traffic.csv',
        'model_output': 'xgboost_modbus_model.pkl'
    },
    'defense': {
        'block_duration': 30,  # Duration to block suspicious IPs
    }
}

# Initialize packet list to capture and store network traffic for analysis
packets = []

# ModbusTCP Packet Construction
def create_modbus_request(transaction_id, function_code, data):
    modbus_header = bytes([transaction_id >> 8, transaction_id & 0xFF,
                           0x00, 0x00,
                           0x00, 0x06,
                           0x01])  # ModbusTCP Header
    modbus_pdu = bytes([function_code]) + bytes(data, 'utf-8')  # Modbus Function Code and Data
    return modbus_header + modbus_pdu  # Return the complete ModbusTCP packet

def preprocess_features(df):
    # Ensure all columns are numerical
    df['tcp_flags'] = pd.Categorical(df['tcp_flags']).codes
    df['ip_len'] = pd.to_numeric(df['ip_len'], errors='coerce').fillna(0)
    df['src_ip'] = df['src_ip'].apply(lambda x: int(ipaddress.ip_address(x)))
    df['dst_ip'] = df['dst_ip'].apply(lambda x: int(ipaddress.ip_address(x)))
    return df

def extract_features(packet, label):
    features = {
        'src_ip': packet[IP].src,
        'dst_ip': packet[IP].dst,
        'tcp_seq': packet[TCP].seq,
        'tcp_ack': packet[TCP].ack,
        'tcp_window_size': packet[TCP].window,
        'tcp_flags': packet[TCP].flags,
        'frame_len': len(packet),
        'ip_len': packet[IP].len,
        'tcp_srcport': packet[TCP].sport,
        'tcp_dstport': packet[TCP].dport,
        'label': label
    }
    return features

# Generate Random Traffic
def generate_random_traffic(is_attack=False):
    tcp_seq = random.randint(1000, 10000)
    tcp_ack = random.randint(1000, 10000)
    tcp_window_size = random.choice([4096, 8192, 16384])
    frame_len = random.randint(59, 200)
    tcp_flags = random.choice([0, 1]) if is_attack else 0
    return tcp_seq, tcp_ack, tcp_window_size, frame_len, tcp_flags

class AnomalyDetector:
    def __init__(self, model_path):
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def detect(self, features):
        df = pd.DataFrame([features])
        df = preprocess_features(df)
        if 'label' in df.columns:
            df = df.drop('label', axis=1)
        prediction = self.model.predict(df)
        return prediction[0] == 1  # Return True for attack, False for normal traffic

class ControlUnit:
    def __init__(self, env, config):
        self.env = env
        self.data = {}
        self.config = config
        self.blocked_ips = {}

    def send_data(self, sensor_name, value):
        self.data[sensor_name] = value

    def actuate(self, actuator_name):
        logging.info(f'{self.env.now}: {actuator_name} actuated')
        print(f'Actuator "{actuator_name}" triggered at time {self.env.now}')

    def mitigate_attack(self, ip):
        if ip not in self.blocked_ips or self.env.now - self.blocked_ips[ip] > self.config['defense']['block_duration']:
            self.blocked_ips[ip] = self.env.now
            logging.warning(f'{self.env.now}: Blocking traffic from {ip} due to suspicious activity.')
            print(f"Mitigating attack: Blocking IP {ip}")

def simulate_normal_modbus_traffic(env, config, control_unit, anomaly_detector):
    while True:
        tcp_seq, tcp_ack, tcp_window_size, frame_len, tcp_flags = generate_random_traffic(is_attack=False)
        packet = IP(src=config['network']['hmi_ip'], dst=config['network']['plc_ip']) / TCP(seq=tcp_seq, ack=tcp_ack, window=tcp_window_size, flags=tcp_flags)
        features = extract_features(packet, label=0)  # Normal traffic labeled as 0
        packets.append(features)  # Capture features for dataset
        yield env.timeout(config['simulation']['normal_traffic_interval'])

def simulate_attack_traffic(env, config, control_unit, anomaly_detector, attack_type):
    while True:
        tcp_seq, tcp_ack, tcp_window_size, frame_len, tcp_flags = generate_random_traffic(is_attack=True)
        if attack_type == "field_flooding":
            packet = IP(src=config['network']['hmi_ip'], dst=config['network']['plc_ip']) / TCP(seq=tcp_seq, ack=tcp_ack, window=tcp_window_size, flags=tcp_flags)
        elif attack_type == "unauthorized_access":
            packet = IP(src=config['network']['hmi_ip'], dst=config['network']['plc_ip']) / TCP(seq=tcp_seq, ack=tcp_ack, window=tcp_window_size, flags=tcp_flags) / create_modbus_request(9999, 0xFF, "Malicious Command")
        elif attack_type == "replay":
            packet = IP(src=config['network']['hmi_ip'], dst=config['network']['plc_ip']) / TCP(seq=tcp_seq, ack=tcp_ack, window=tcp_window_size, flags=tcp_flags)
        elif attack_type == "dos":
            packet = IP(src=config['network']['hmi_ip'], dst=config['network']['plc_ip']) / TCP(seq=tcp_seq, ack=tcp_ack, window=tcp_window_size, flags=tcp_flags)

        features = extract_features(packet, label=1)  # Attack labeled as 1
        if anomaly_detector.detect(features):
            logging.warning(f'{env.now}: {attack_type.capitalize()} Attack detected!')
            control_unit.mitigate_attack(config['network']['plc_ip'])
        packets.append(features)
        yield env.timeout(config['simulation']['attack_interval'])

def evaluate_model(packets_df):
    packets_df = preprocess_features(packets_df)
    X = packets_df.drop(columns=['label'])
    y = packets_df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    joblib.dump(model, config['paths']['model_output'])

def run_simulation():
    env = simpy.Environment()
    control_unit = ControlUnit(env, config)
    anomaly_detector = AnomalyDetector(config['paths']['model_output'])

    # Normal traffic generation
    for _ in range(10):
        env.process(simulate_normal_modbus_traffic(env, config, control_unit, anomaly_detector))
    
    # Attack traffic generation
    attack_types = ["field_flooding", "unauthorized_access", "replay", "dos"]
    for attack_type in attack_types:
        env.process(simulate_attack_traffic(env, config, control_unit, anomaly_detector, attack_type))

    env.run(until=config['simulation']['duration'])
    
    # Convert packet features to DataFrame and save to CSV for training
    packets_df = pd.DataFrame(packets)
    packets_df.to_csv(config['paths']['dataset_output'], index=False)
    evaluate_model(packets_df)

if __name__ == "__main__":
    run_simulation()


# In[ ]:


#using hyper parameter


# In[5]:


import simpy
import random
import ipaddress
import scapy.all as scapy
import logging
from xgboost import XGBClassifier, plot_importance
import numpy as np
import pandas as pd
import joblib
from scapy.layers.inet import IP, TCP
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import yaml
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s: %(levelname)s: %(message)s')

# Configuration for simulation
config = {
    'simulation': {
        'duration': 2500,  # Total duration of the simulation
        'normal_traffic_interval': 2,  # Interval between normal packets
        'attack_interval': 5,  # Interval between attack packets
    },
    'network': {
        'hmi_ip': '192.168.1.100',
        'plc_ip': '192.168.1.200',
        'modbus_port': 502,
    },
    'paths': {
        'dataset_output': 'balanced_modbus_traffic.csv',
        'model_output': 'xgboost_modbus_model.pkl'
    },
    'defense': {
        'block_duration': 30,  # Duration to block suspicious IPs
    }
}

# Initialize packet list to capture and store network traffic for analysis
packets = []

# ModbusTCP Packet Construction
def create_modbus_request(transaction_id, function_code, data):
    modbus_header = bytes([transaction_id >> 8, transaction_id & 0xFF,
                           0x00, 0x00,
                           0x00, 0x06,
                           0x01])  # ModbusTCP Header
    modbus_pdu = bytes([function_code]) + bytes(data, 'utf-8')  # Modbus Function Code and Data
    return modbus_header + modbus_pdu  # Return the complete ModbusTCP packet

def preprocess_features(df):
    # Convert 'tcp_flags' to categorical codes if necessary
    if df['tcp_flags'].dtype == 'object':
        df['tcp_flags'] = pd.Categorical(df['tcp_flags']).codes

    # Convert 'ip_len' to numeric
    df['ip_len'] = pd.to_numeric(df['ip_len'], errors='coerce').fillna(0)

    # Convert IP addresses to a numerical representation
    df['src_ip'] = df['src_ip'].apply(lambda x: int(ipaddress.ip_address(x)))
    df['dst_ip'] = df['dst_ip'].apply(lambda x: int(ipaddress.ip_address(x)))

    # Ensure all other columns are numerical
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df

def extract_features(packet, label):
    features = {
        'src_ip': packet[IP].src,  # Add source IP
        'dst_ip': packet[IP].dst,  # Add destination IP
        'tcp_seq': packet[TCP].seq,
        'tcp_ack': packet[TCP].ack,
        'tcp_window_size': packet[TCP].window,
        'tcp_flags': packet[TCP].flags,
        'frame_len': len(packet),
        'ip_len': packet[IP].len,
        'tcp_srcport': packet[TCP].sport,
        'tcp_dstport': packet[TCP].dport,
        'label': label
    }
    return features

# Generate Random Traffic (Both Normal and Attack Traffic)
def generate_random_traffic(is_attack=False):
    tcp_seq = random.randint(1000, 10000)
    tcp_ack = random.randint(1000, 10000)
    tcp_window_size = random.choice([4096, 8192, 16384])  # Randomize common window sizes
    frame_len = random.randint(59, 200)  # Vary packet size
    tcp_flags = random.choice([0, 1]) if is_attack else 0  # Randomize flags for attack packets
    return tcp_seq, tcp_ack, tcp_window_size, frame_len, tcp_flags

# Anomaly Detector using XGBoost
class AnomalyDetector:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def detect(self, features):
        df = pd.DataFrame([features])
        df = preprocess_features(df)
        if 'label' in df.columns:
            df = df.drop('label', axis=1)
        prediction = self.model.predict(df)
        return prediction[0] == 1  # Return True for attack, False for normal traffic

# Control Unit for Mitigating Attacks
class ControlUnit:
    def __init__(self, env, config):
        self.env = env
        self.data = {}
        self.config = config
        self.blocked_ips = {}

    def send_data(self, sensor_name, value):
        self.data[sensor_name] = value

    def actuate(self, actuator_name):
        logging.info(f'{self.env.now}: {actuator_name} actuated')
        print(f'Actuator "{actuator_name}" triggered at time {self.env.now}')

    def mitigate_attack(self, ip):
        if ip not in self.blocked_ips or self.env.now - self.blocked_ips[ip] > self.config['defense']['block_duration']:
            self.blocked_ips[ip] = self.env.now
            logging.warning(f'{self.env.now}: Blocking traffic from {ip} due to suspicious activity.')
            print(f"Mitigating attack: Blocking IP {ip}")


def simulate_normal_modbus_traffic(env, config, control_unit, anomaly_detector):
    while True:
        tcp_seq, tcp_ack, tcp_window_size, frame_len, tcp_flags = generate_random_traffic(is_attack=False)
        packet = IP(src=config['network']['hmi_ip'], dst=config['network']['plc_ip']) / TCP(seq=tcp_seq, ack=tcp_ack, window=tcp_window_size, flags=tcp_flags)
        features = extract_features(packet, label=0)  # Normal traffic labeled as 0
        packets.append(features)  # Capture features for dataset
        yield env.timeout(config['simulation']['normal_traffic_interval'])

def simulate_attack_traffic(env, config, control_unit, anomaly_detector, attack_type):
    while True:
        tcp_seq, tcp_ack, tcp_window_size, frame_len, tcp_flags = generate_random_traffic(is_attack=True)
        if attack_type == "field_flooding":
            packet = IP(src=config['network']['hmi_ip'], dst=config['network']['plc_ip']) / TCP(seq=tcp_seq, ack=tcp_ack, window=tcp_window_size, flags=tcp_flags)
        elif attack_type == "unauthorized_access":
            packet = IP(src=config['network']['hmi_ip'], dst=config['network']['plc_ip']) / TCP(seq=tcp_seq, ack=tcp_ack, window=tcp_window_size, flags=tcp_flags) / create_modbus_request(9999, 0xFF, "Malicious Command")
        elif attack_type == "replay":
            packet = IP(src=config['network']['hmi_ip'], dst=config['network']['plc_ip']) / TCP(seq=tcp_seq, ack=tcp_ack, window=tcp_window_size, flags=tcp_flags)
        elif attack_type == "dos":
            packet = IP(src=config['network']['hmi_ip'], dst=config['network']['plc_ip']) / TCP(seq=tcp_seq, ack=tcp_ack, window=tcp_window_size, flags=tcp_flags)

        features = extract_features(packet, label=1)  # Attack labeled as 1
        if anomaly_detector.detect(features):
            logging.warning(f'{env.now}: {attack_type.capitalize()} Attack detected!')
            control_unit.mitigate_attack(config['network']['plc_ip'])
        packets.append(features)
        yield env.timeout(config['simulation']['attack_interval'])

# Evaluate model accuracy
def evaluate_model(packets_df):
    # Preprocess features and labels
    packets_df = preprocess_features(packets_df)
    X = packets_df.drop(columns=['label'])
    y = packets_df['label']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameter tuning using RandomizedSearchCV
    xgb_model = XGBClassifier( eval_metric='logloss')
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.1],
        'max_depth': [3, 5, 7, 10],
        'n_estimators': [100, 200, 300],
        'subsample': [0.5, 0.7, 0.8, 1.0],
        'gamma': [0, 0.1, 0.5, 1],
        'colsample_bytree': [0.5, 0.7, 1.0],
        'scale_pos_weight': [1, 5, 10]  # Adjust this based on class imbalance
    }

    random_search = RandomizedSearchCV(xgb_model, param_distributions=param_grid, n_iter=50, scoring='f1', cv=5, random_state=42)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    
    # Predictions and accuracy score
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    
    # Visualize feature importance
    plot_importance(best_model)
    plt.title('Feature Importance')
    plt.show()

# Main simulation function
def main():
    env = simpy.Environment()
    anomaly_detector = AnomalyDetector(config['paths']['model_output'])
    control_unit = ControlUnit(env, config)
    
    env.process(simulate_normal_modbus_traffic(env, config, control_unit, anomaly_detector))
    env.process(simulate_attack_traffic(env, config, control_unit, anomaly_detector, attack_type="unauthorized_access"))

    # Run the simulation
    env.run(until=config['simulation']['duration'])

    # Save dataset and evaluate model
    packets_df = pd.DataFrame(packets)
    packets_df.to_csv(config['paths']['dataset_output'], index=False)
    evaluate_model(packets_df)

if __name__ == "__main__":
    main()


# In[ ]:




