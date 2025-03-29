import os
import random
import uuid
from datetime import datetime
from textwrap import dedent


def create_directory():
    """Create a directory for generated files if it doesn't exist"""
    dir_name = "generated_programs"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def generate_program_1(dir_name):
    """Generate a machine learning program with sklearn"""
    filename = os.path.join(dir_name, "ml_classifier.py")
    content = dedent(f'''\
    ''')
    # Advanced Machine Learning Classifier
    # Generated on {datetime.now()}
    # This program demonstrates a complete ML pipeline
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    
    # Generate synthetic data
    np.random.seed(42)
    data_size = 1000
    X = np.random.randn(data_size, 10)
    y = np.random.randint(0, 2, data_size)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
        print("\\nFeature Importances:")
        importances = pipeline.named_steps['classifier'].feature_importances_
        for i, imp in enumerate(importances):
            print(f"Feature {{i}}: {{imp:.4f}}")
    
    # Save model (commented out for this example)
    # import joblib
    # joblib.dump(pipeline, 'model.pkl')
    ''')

    with open(filename, 'w') as f:
        f.write(content)


def generate_program_2(dir_name):
    """Generate a web scraper using BeautifulSoup"""
    filename = os.path.join(dir_name, "web_scraper.py")
    content = dedent(f'''\
    # Advanced Web Scraper
    # Generated on {datetime.now()}
    # This program scrapes data from a website and saves to CSV
    
    import requests
    from bs4 import BeautifulSoup
    import csv
    import time
    from urllib.parse import urljoin
    from fake_useragent import UserAgent
    
    class WebScraper:
        def __init__(self, base_url):
            self.base_url = base_url
            self.ua = UserAgent()
            self.session = requests.Session()
            self.data = []
            
        def get_random_headers(self):
            return {{
                'User-Agent': self.ua.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.google.com/',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }}
            
        def scrape_page(self, url):
            try:
                headers = self.get_random_headers()
                response = self.session.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Example: Extract article titles and links
                articles = soup.find_all('article', limit=5)
                for article in articles:
                    title = article.find('h2').get_text(strip=True) if article.find('h2') else 'N/A'
                    link = article.find('a')['href'] if article.find('a') else 'N/A'
                    if link and not link.startswith('http'):
                        link = urljoin(self.base_url, link)
                    
                    self.data.append({{
                        'title': title,
                        'link': link,
                        'timestamp': datetime.now().isoformat()
                    }})
                
                time.sleep(random.uniform(1, 3))  # Polite delay
                
            except Exception as e:
                print(f"Error scraping {{url}}: {{str(e)}}")
                
        def save_to_csv(self, filename):
            if not self.data:
                print("No data to save")
                return
                
            keys = self.data[0].keys()
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.data)
            print(f"Data saved to {{filename}}")
    
    if __name__ == '__main__':
        # Example usage (commented out to prevent accidental runs)
        # scraper = WebScraper('https://example.com/news')
        # scraper.scrape_page(scraper.base_url)
        # scraper.save_to_csv('scraped_data.csv')
        print("WebScraper class ready. Uncomment code to run.")
    ''')

    with open(filename, 'w') as f:
        f.write(content)


def generate_program_3(dir_name):
    """Generate a Flask web application"""
    filename = os.path.join(dir_name, "flask_app.py")
    content = dedent(f'''\
    # Advanced Flask Web Application
    # Generated on {datetime.now()}
    # This program creates a RESTful API with Flask
    
    from flask import Flask, request, jsonify
    from flask_sqlalchemy import SQLAlchemy
    from flask_marshmallow import Marshmallow
    from flask_cors import CORS
    import os
    from datetime import datetime
    
    # Initialize app
    app = Flask(__name__)
    CORS(app)
    
    # Database configuration
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'db.sqlite')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize database
    db = SQLAlchemy(app)
    
    # Initialize Marshmallow
    ma = Marshmallow(app)
    
    # Product Model
    class Product(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.String(100), unique=True)
        description = db.Column(db.String(200))
        price = db.Column(db.Float)
        qty = db.Column(db.Integer)
        created_at = db.Column(db.DateTime, default=datetime.utcnow)
        
        def __init__(self, name, description, price, qty):
            self.name = name
            self.description = description
            self.price = price
            self.qty = qty
    
    # Product Schema
    class ProductSchema(ma.Schema):
        class Meta:
            fields = ('id', 'name', 'description', 'price', 'qty', 'created_at')
    
    # Initialize schema
    product_schema = ProductSchema()
    products_schema = ProductSchema(many=True)
    
    # Routes
    @app.route('/product', methods=['POST'])
    def add_product():
        name = request.json['name']
        description = request.json['description']
        price = request.json['price']
        qty = request.json['qty']
        
        new_product = Product(name, description, price, qty)
        
        db.session.add(new_product)
        db.session.commit()
        
        return product_schema.jsonify(new_product)
    
    @app.route('/product', methods=['GET'])
    def get_products():
        all_products = Product.query.all()
        result = products_schema.dump(all_products)
        return jsonify(result)
    
    @app.route('/product/<id>', methods=['GET'])
    def get_product(id):
        product = Product.query.get(id)
        return product_schema.jsonify(product)
    
    @app.route('/product/<id>', methods=['PUT'])
    def update_product(id):
        product = Product.query.get(id)
        
        name = request.json['name']
        description = request.json['description']
        price = request.json['price']
        qty = request.json['qty']
        
        product.name = name
        product.description = description
        product.price = price
        product.qty = qty
        
        db.session.commit()
        
        return product_schema.jsonify(product)
    
    @app.route('/product/<id>', methods=['DELETE'])
    def delete_product(id):
        product = Product.query.get(id)
        db.session.delete(product)
        db.session.commit()
        
        return product_schema.jsonify(product)
    
    # Run Server
    if __name__ == '__main__':
        with app.app_context():
            db.create_all()
        app.run(debug=True)
    ''')

    with open(filename, 'w') as f:
        f.write(content)


def generate_program_4(dir_name):
    """Generate a data analysis program with pandas and matplotlib"""
    filename = os.path.join(dir_name, "data_analysis.py")
    content = dedent(f'''\
    # Advanced Data Analysis Program
    # Generated on {datetime.now()}
    # This program demonstrates data cleaning, analysis, and visualization
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic dataset
    def generate_data(num_samples=1000):
        data = {{
            'age': np.random.normal(45, 15, num_samples).astype(int),
            'income': np.random.lognormal(4, 0.4, num_samples).astype(int),
            'education': np.random.choice(
                ['High School', 'Bachelor', 'Master', 'PhD'],
                num_samples,
                p=[0.3, 0.4, 0.2, 0.1]
            ),
            'satisfaction': np.random.randint(1, 11, num_samples),
            'region': np.random.choice(
                ['North', 'South', 'East', 'West'],
                num_samples
            ),
            'purchases': np.random.poisson(5, num_samples)
        }}
        return pd.DataFrame(data)
    
    # Load data
    df = generate_data()
    
    # Data cleaning
    def clean_data(df):
        # Handle negative ages (artifacts from normal distribution)
        df['age'] = df['age'].apply(lambda x: x if x > 0 else 0)
        
        # Cap income at 99th percentile to handle extreme values
        income_99 = df['income'].quantile(0.99)
        df['income'] = df['income'].apply(lambda x: min(x, income_99))
        
        return df
    
    df = clean_data(df)
    
    # Exploratory Data Analysis
    def perform_eda(df):
        print("\\n=== Basic Statistics ===")
        print(df.describe())
        
        print("\\n=== Categorical Value Counts ===")
        for col in ['education', 'region']:
            print(f"\\n{{col}}:")
            print(df[col].value_counts())
        
        print("\\n=== Correlation Matrix ===")
        numeric_cols = df.select_dtypes(include=np.number).columns
        corr_matrix = df[numeric_cols].corr()
        print(corr_matrix)
        
        # Visualizations
        plt.figure(figsize=(12, 8))
        
        # Age distribution
        plt.subplot(2, 2, 1)
        sns.histplot(df['age'], kde=True)
        plt.title('Age Distribution')
        
        # Income distribution
        plt.subplot(2, 2, 2)
        sns.histplot(df['income'], kde=True)
        plt.title('Income Distribution')
        
        # Satisfaction by education
        plt.subplot(2, 2, 3)
        sns.boxplot(x='education', y='satisfaction', data=df, 
                   order=['High School', 'Bachelor', 'Master', 'PhD'])
        plt.title('Satisfaction by Education Level')
        plt.xticks(rotation=45)
        
        # Purchases by region
        plt.subplot(2, 2, 4)
        sns.barplot(x='region', y='purchases', data=df, ci=None)
        plt.title('Average Purchases by Region')
        
        plt.tight_layout()
        plt.savefig('eda_visualizations.png')
        print("\\nEDA visualizations saved to eda_visualizations.png")
    
    perform_eda(df)
    
    # Hypothesis testing example
    def test_hypothesis(df):
        print("\\n=== Hypothesis Testing ===")
        
        # Compare satisfaction between Bachelor and Master degree holders
        bachelor = df[df['education'] == 'Bachelor']['satisfaction']
        master = df[df['education'] == 'Master']['satisfaction']
        
        t_stat, p_value = stats.ttest_ind(bachelor, master)
        print(f"T-test between Bachelor and Master satisfaction:")
        print(f"T-statistic: {{t_stat:.4f}}, P-value: {{p_value:.4f}}")
        
        if p_value < 0.05:
            print("Significant difference detected (p < 0.05)")
        else:
            print("No significant difference detected (p >= 0.05)")
    
    test_hypothesis(df)
    ''')

    with open(filename, 'w') as f:
        f.write(content)


def generate_program_5(dir_name):
    """Generate a text-based adventure game"""
    filename = os.path.join(dir_name, "text_adventure.py")
    content = dedent(f'''\
    # Advanced Text Adventure Game
    # Generated on {datetime.now()}
    # This program implements a complete text adventure with inventory, combat, and multiple endings
    
    import random
    import time
    import sys
    from enum import Enum
    
    class Direction(Enum):
        NORTH = 'north'
        SOUTH = 'south'
        EAST = 'east'
        WEST = 'west'
    
    class RoomType(Enum):
        NORMAL = 'normal'
        TREASURE = 'treasure'
        ENEMY = 'enemy'
        TRAP = 'trap'
        EXIT = 'exit'
    
    class Item:
        def __init__(self, name, description, usable=False, damage=0, health=0):
            self.name = name
            self.description = description
            self.usable = usable
            self.damage = damage
            self.health = health
        
        def __str__(self):
            return f"{{self.name}}: {{self.description}}"
    
    class Enemy:
        def __init__(self, name, health, damage, description):
            self.name = name
            self.health = health
            self.damage = damage
            self.description = description
        
        def is_alive(self):
            return self.health > 0
    
    class Player:
        def __init__(self):
            self.health = 100
            self.inventory = []
            self.current_room = None
            self.score = 0
            self.game_over = False
        
        def add_item(self, item):
            self.inventory.append(item)
            print(f"You picked up {{item.name}}")
        
        def use_item(self, item_name):
            for item in self.inventory:
                if item.name.lower() == item_name.lower():
                    if item.usable:
                        self.health += item.health
                        print(f"You used {{item.name}} and gained {{item.health}} health")
                        self.inventory.remove(item)
                        return True
            print(f"You don't have a usable {{item_name}}")
            return False
        
        def show_inventory(self):
            if not self.inventory:
                print("Your inventory is empty")
                return
            
            print("Inventory:")
            for item in self.inventory:
                print(f"- {{item}}")
        
        def attack(self, enemy, item=None):
            if item:
                damage = item.damage
                print(f"You attack {{enemy.name}} with {{item.name}} for {{damage}} damage!")
            else:
                damage = random.randint(5, 15)
                print(f"You punch {{enemy.name}} for {{damage}} damage!")
            
            enemy.health -= damage
            
            if enemy.is_alive():
                player_damage = random.randint(5, enemy.damage)
                self.health -= player_damage
                print(f"{{enemy.name}} hits you for {{player_damage}} damage!")
                
                if self.health <= 0:
                    print("\\nYou have been defeated!")
                    self.game_over = True
            else:
                print(f"You defeated {{enemy.name}}!")
                self.score += 50
    
    class Room:
        def __init__(self, name, description, room_type=RoomType.NORMAL):
            self.name = name
            self.description = description
            self.room_type = room_type
            self.connections = {{}}
            self.items = []
            self.enemy = None
            self.visited = False
        
        def add_connection(self, direction, room):
            self.connections[direction] = room
        
        def add_item(self, item):
            self.items.append(item)
        
        def add_enemy(self, enemy):
            self.enemy = enemy
        
        def enter(self, player):
            player.current_room = self
            
            if not self.visited:
                print(f"\\n{{self.name}}")
                print(self.description)
                self.visited = True
            else:
                print(f"\\n{{self.name}} (visited)")
            
            if self.room_type == RoomType.TREASURE and not self.visited:
                treasure = random.choice([
                    Item("Golden Chalice", "A priceless artifact", False),
                    Item("Health Potion", "Restores 30 health", True, 0, 30),
                    Item("Magic Sword", "A sword that glows with power", False, 25)
                ])
                self.add_item(treasure)
                print("You spot a treasure chest in the corner!")
            
            if self.items:
                print("\\nYou see the following items:")
                for item in self.items:
                    print(f"- {{item}}")
            
            if self.enemy and self.enemy.is_alive():
                print(f"\\nDanger! {{self.enemy.description}}")
            
            if self.room_type == RoomType.EXIT:
                print("\\nYou see the exit to the dungeon!")
                print("Congratulations! You escaped!")
                player.score += 1000
                player.game_over = True
            
            if self.room_type == RoomType.TRAP and not self.visited:
                trap_damage = random.randint(10, 30)
                player.health -= trap_damage
                print(f"\\nYou triggered a trap and took {{trap_damage}} damage!")
                if player.health <= 0:
                    print("\\nYou have been defeated by the trap!")
                    player.game_over = True
    
    class Game:
        def __init__(self):
            self.player = Player()
            self.rooms = self.create_world()
            self.player.current_room = self.rooms['entrance']
        
        def create_world(self):
            rooms = {{}}
            
            # Create rooms
            rooms['entrance'] = Room(
                "Dungeon Entrance",
                "You stand at the entrance of a dark dungeon. Torches flicker on the walls.",
                RoomType.NORMAL
            )
            
            rooms['hallway'] = Room(
                "Long Hallway",
                "A long hallway stretches before you. The stones are damp and cold.",
                RoomType.NORMAL
            )
            
            rooms['treasure'] = Room(
                "Treasure Room",
                "A room filled with glittering gold and precious gems.",
                RoomType.TREASURE
            )
            
            rooms['armory'] = Room(
                "Ancient Armory",
                "Rusty weapons line the walls of this old armory.",
                RoomType.NORMAL
            )
            
            rooms['spider'] = Room(
                "Spider's Lair",
                "Thick webs cover the walls. Something moves in the shadows...",
                RoomType.ENEMY
            )
            
            rooms['trap'] = Room(
                "Dark Passage",
                "A narrow passage with suspicious looking tiles on the floor.",
                RoomType.TRAP
            )
            
            rooms['exit'] = Room(
                "Dungeon Exit",
                "Sunlight filters in through the exit.",
                RoomType.EXIT
            )
            
            # Add connections
            rooms['entrance'].add_connection(Direction.EAST, rooms['hallway'])
            rooms['hallway'].add_connection(Direction.WEST, rooms['entrance'])
            rooms['hallway'].add_connection(Direction.NORTH, rooms['treasure'])
            rooms['hallway'].add_connection(Direction.SOUTH, rooms['trap'])
            rooms['hallway'].add_connection(Direction.EAST, rooms['spider'])
            rooms['treasure'].add_connection(Direction.SOUTH, rooms['hallway'])
            rooms['trap'].add_connection(Direction.NORTH, rooms['hallway'])
            rooms['spider'].add_connection(Direction.WEST, rooms['hallway'])
            rooms['spider'].add_connection(Direction.EAST, rooms['armory'])
            rooms['armory'].add_connection(Direction.WEST, rooms['spider'])
            rooms['armory'].add_connection(Direction.SOUTH, rooms['exit'])
            rooms['exit'].add_connection(Direction.NORTH, rooms['armory'])
            
            # Add items
            rooms['armory'].add_item(Item("Rusty Sword", "An old but still sharp sword", False, 15))
            rooms['armory'].add_item(Item("Wooden Shield", "Provides some protection", False))
            
            # Add enemies
            rooms['spider'].add_enemy(Enemy(
                "Giant Spider",
                50,
                10,
                "A massive spider with glowing red eyes!"
            ))
            
            return rooms
        
        def handle_command(self, command):
            command = command.lower().strip()
            
            if command in ['n', 'north']:
                self.move_player(Direction.NORTH)
            elif command in ['s', 'south']:
                self.move_player(Direction.SOUTH)
            elif command in ['e', 'east']:
                self.move_player(Direction.EAST)
            elif command in ['w', 'west']:
                self.move_player(Direction.WEST)
            elif command.startswith('take '):
                item_name = command[5:]
                self.take_item(item_name)
            elif command.startswith('use '):
                item_name = command[4:]
                self.player.use_item(item_name)
            elif command == 'inventory':
                self.player.show_inventory()
            elif command.startswith('attack'):
                self.attack_enemy()
            elif command in ['quit', 'exit']:
                print("Thanks for playing!")
                self.player.game_over = True
            elif command == 'help':
                self.show_help()
            else:
                print("I don't understand that command. Type 'help' for options.")
        
        def move_player(self, direction):
            if direction in self.player.current_room.connections:
                if (self.player.current_room.enemy and 
                    self.player.current_room.enemy.is_alive()):
                    print(f"You can't leave while {{self.player.current_room.enemy.name}} is attacking!")
                    return
                
                self.player.current_room = self.player.current_room.connections[direction]
                self.player.current_room.enter(self.player)
            else:
                print("You can't go that way!")
        
        def take_item(self, item_name):
            room = self.player.current_room
            for item in room.items:
                if item.name.lower() == item_name.lower():
                    self.player.add_item(item)
                    room.items.remove(item)
                    return
            
            print(f"You don't see {{item_name}} here")
        
        def attack_enemy(self):
            room = self.player.current_room
            if room.enemy and room.enemy.is_alive():
                weapon = None
                for item in self.player.inventory:
                    if item.damage > 0:
                        weapon = item
                        break
                
                self.player.attack(room.enemy, weapon)
            else:
                print("There's nothing to attack here")
        
        def show_help(self):
            print("\\nAvailable commands:")
            print("north/n, south/s, east/e, west/w - Move in that direction")
            print("take <item> - Pick up an item")
            print("use <item> - Use a consumable item")
            print("attack - Attack an enemy")
            print("inventory - Show your inventory")
            print("help - Show this help message")
            print("quit/exit - Quit the game")
        
        def start(self):
            print("=== DUNGEON ADVENTURE ===")
            print("Type 'help' for commands\\n")
            
            self.player.current_room.enter(self.player)
            
            while not self.player.game_over:
                command = input("\\nWhat would you like to do? ")
                self.handle_command(command)
            
            print(f"\\nGame over! Your score: {{self.player.score}}")
    
    if __name__ == '__main__':
        game = Game()
        game.start()
    ''')

    with open(filename, 'w') as f:
        f.write(content)


def generate_program_6(dir_name):
    """Generate a multithreaded network server"""
    filename = os.path.join(dir_name, "network_server.py")
    content = dedent(f'''\
    # Advanced Multithreaded Network Server
    # Generated on {datetime.now()}
    # This program implements a TCP server with threading and custom protocol
    
    import socket
    import threading
    import time
    import json
    from datetime import datetime
    
    class ThreadedServer:
        def __init__(self, host, port):
            self.host = host
            self.port = port
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.host, self.port))
            self.clients = {}
            self.running = False
            self.command_handlers = {
                'ECHO': self.handle_echo,
                'TIME': self.handle_time,
                'CLIENTS': self.handle_clients,
                'UPPER': self.handle_upper,
                'LOWER': self.handle_lower,
                'CALC': self.handle_calc,
                'QUIT': self.handle_quit
            }
        
        def listen(self):
            self.sock.listen(5)
            self.running = True
            print(f"Server listening on {{self.host}}:{{self.port}}")
            
            accept_thread = threading.Thread(target=self.accept_connections)
            accept_thread.daemon = True
            accept_thread.start()
            
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\\nShutting down server...")
                self.shutdown()
        
        def accept_connections(self):
            while self.running:
                try:
                    client, address = self.sock.accept()
                    client.settimeout(60)
                    
                    client_id = str(len(self.clients) + 1)
                    self.clients[client_id] = {
                        'connection': client,
                        'address': address,
                        'active': True
                    }
                    
                    print(f"New connection from {{address}} (ID: {{client_id}})")
                    
                    thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_id,)
                    )
                    thread.daemon = True
                    thread.start()
                
                except Exception as e:
                    if self.running:
                        print(f"Error accepting connection: {{e}}")
        
        def handle_client(self, client_id):
            client_info = self.clients[client_id]
            client = client_info['connection']
            address = client_info['address']
            
            try:
                while client_info['active'] and self.running:
                    try:
                        data = client.recv(4096).decode('utf-8')
                        if not data:
                            break
                        
                        print(f"Received from {{address}} (ID: {{client_id}}): {{data.strip()}}")
                        
                        try:
                            request = json.loads(data)
                            command = request.get('command', '').upper()
                            args = request.get('args', [])
                            
                            if command in self.command_handlers:
                                response = self.command_handlers[command](client_id, *args)
                            else:
                                response = {
                                    'status': 'error',
                                    'message': 'Unknown command'
                                }
                        
                        except json.JSONDecodeError:
                            response = {
                                'status': 'error',
                                'message': 'Invalid JSON format'
                            }
                        
                        client.send(json.dumps(response).encode('utf-8'))
                    
                    except socket.timeout:
                        print(f"Client {{address}} (ID: {{client_id}}) timeout")
                        break
                    
                    except Exception as e:
                        print(f"Error with client {{address}} (ID: {{client_id}}): {{e}}")
                        break
            
            finally:
                client.close()
                if client_id in self.clients:
                    del self.clients[client_id]
                print(f"Connection closed for {{address}} (ID: {{client_id}})")
        
        def handle_echo(self, client_id, *args):
            return {
                'status': 'success',
                'message': ' '.join(args)
            }
        
        def handle_time(self, client_id, *args):
            return {
                'status': 'success',
                'time': datetime.now().isoformat()
            }
        
        def handle_clients(self, client_id, *args):
            active_clients = [
                {'id': cid, 'address': str(info['address'])}
                for cid, info in self.clients.items()
                if info['active']
            ]
            return {
                'status': 'success',
                'clients': active_clients,
                'count': len(active_clients)
            }
        
        def handle_upper(self, client_id, *args):
            if not args:
                return {
                    'status': 'error',
                    'message': 'No text provided'
                }
            return {
                'status': 'success',
                'result': ' '.join(args).upper()
            }
        
        def handle_lower(self, client_id, *args):
            if not args:
                return {
                    'status': 'error',
                    'message': 'No text provided'
                }
            return {
                'status': 'success',
                'result': ' '.join(args).lower()
            }
        
        def handle_calc(self, client_id, *args):
            if len(args) != 3:
                return {
                    'status': 'error',
                    'message': 'Usage: CALC <num1> <+|-|*|/> <num2>'
                }
            
            try:
                num1 = float(args[0])
                op = args[1]
                num2 = float(args[2])
                
                if op == '+':
                    result = num1 + num2
                elif op == '-':
                    result = num1 - num2
                elif op == '*':
                    result = num1 * num2
                elif op == '/':
                    if num2 == 0:
                        return {
                            'status': 'error',
                            'message': 'Division by zero'
                        }
                    result = num1 / num2
                else:
                    return {
                        'status': 'error',
                        'message': 'Invalid operator'
                    }
                
                return {
                    'status': 'success',
                    'result': result
                }
            
            except ValueError:
                return {
                    'status': 'error',
                    'message': 'Invalid numbers'
                }
        
        def handle_quit(self, client_id, *args):
            if client_id in self.clients:
                self.clients[client_id]['active'] = False
            return {
                'status': 'success',
                'message': 'Goodbye!'
            }
        
        def shutdown(self):
            self.running = False
            for client_id, info in self.clients.items():
                info['active'] = False
                info['connection'].close()
            self.sock.close()
    
    if __name__ == '__main__':
        server = ThreadedServer('127.0.0.1', 65432)
        server.listen()
    ''')

    with open(filename, 'w') as f:
        f.write(content)


def generate_program_7(dir_name):
    """Generate a password manager with encryption"""
    filename = os.path.join(dir_name, "password_manager.py")
    content = dedent(f'''\
    # Advanced Password Manager with Encryption
    # Generated on {datetime.now()}
    # This program securely stores and retrieves passwords using AES encryption
    
    import os
    import json
    import base64
    import getpass
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    
    class PasswordManager:
        def __init__(self, data_file='passwords.dat', key_file='secret.key'):
            self.data_file = data_file
            self.key_file = key_file
            self.fernet = None
            self.passwords = {}
        
        def generate_key(self, password, salt=None):
            if salt is None:
                salt = os.urandom(16)
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            return key, salt
        
        def initialize(self):
            if os.path.exists(self.key_file):
                print("Key file already exists. Loading existing key.")
                return
            
            password = getpass.getpass("Enter a master password: ")
            confirm = getpass.getpass("Confirm master password: ")
            
            if password != confirm:
                print("Passwords don't match!")
                return False
            
            key, salt = self.generate_key(password)
            
            with open(self.key_file, 'wb') as f:
                f.write(salt + b'::' + key)
            
            print("Password manager initialized successfully.")
            return True
        
        def load_key(self):
            if not os.path.exists(self.key_file):
                print("Key file not found. Please initialize first.")
                return False
            
            password = getpass.getpass("Enter master password: ")
            
            with open(self.key_file, 'rb') as f:
                salt, stored_key = f.read().split(b'::')
            
            try:
                key, _ = self.generate_key(password, salt)
                if key != stored_key:
                    print("Invalid password!")
                    return False
                
                self.fernet = Fernet(key)
                return True
            
            except Exception as e:
                print(f"Error loading key: {{e}}")
                return False
        
        def load_passwords(self):
            if not os.path.exists(self.data_file):
                self.passwords = {}
                return True
            
            if not self.fernet:
                print("Key not loaded")
                return False
            
            try:
                with open(self.data_file, 'rb') as f:
                    encrypted_data = f.read()
                
                decrypted_data = self.fernet.decrypt(encrypted_data)
                self.passwords = json.loads(decrypted_data.decode())
                return True
            
            except Exception as e:
                print(f"Error loading passwords: {{e}}")
                return False
        
        def save_passwords(self):
            if not self.fernet:
                print("Key not loaded")
                return False
            
            try:
                json_data = json.dumps(self.passwords).encode()
                encrypted_data = self.fernet.encrypt(json_data)
                
                with open(self.data_file, 'wb') as f:
                    f.write(encrypted_data)
                
                return True
            
            except Exception as e:
                print(f"Error saving passwords: {{e}}")
                return False
        
        def add_password(self, service, username, password):
            if service in self.passwords:
                print(f"Service {{service}} already exists. Use update instead.")
                return False
            
            self.passwords[service] = {
                'username': username,
                'password': password,
                'created': datetime.now().isoformat(),
                'updated': datetime.now().isoformat()
            }
            
            if self.save_passwords():
                print(f"Password for {{service}} added successfully.")
                return True
            return False
        
        def get_password(self, service):
            if service not in self.passwords:
                print(f"Service {{service}} not found.")
                return None
            
            return self.passwords[service]
        
        def update_password(self, service, new_password=None, new_username=None):
            if service not in self.passwords:
                print(f"Service {{service}} not found.")
                return False
            
            if new_username:
                self.passwords[service]['username'] = new_username
            if new_password:
                self.passwords[service]['password'] = new_password
            
            self.passwords[service]['updated'] = datetime.now().isoformat()
            
            if self.save_passwords():
                print(f"Password for {{service}} updated successfully.")
                return True
            return False
        
        def delete_password(self, service):
            if service not in self.passwords:
                print(f"Service {{service}} not found.")
                return False
            
            del self.passwords[service]
            
            if self.save_passwords():
                print(f"Password for {{service}} deleted successfully.")
                return True
            return False
        
        def list_services(self):
            if not self.passwords:
                print("No passwords stored.")
                return
            
            print("\\nStored services:")
            for i, service in enumerate(self.passwords.keys(), 1):
                print(f"{{i}}. {{service}}")
            print()
        
        def interactive_menu(self):
            if not self.load_key():
                return
            
            if not self.load_passwords():
                return
            
            while True:
                print("\\n=== Password Manager ===")
                print("1. Add new password")
                print("2. Get password")
                print("3. Update password")
                print("4. Delete password")
                print("5. List all services")
                print("6. Exit")
                
                choice = input("Enter your choice (1-6): ")
                
                if choice == '1':
                    service = input("Enter service name: ")
                    username = input("Enter username: ")
                    password = getpass.getpass("Enter password: ")
                    self.add_password(service, username, password)
                
                elif choice == '2':
                    service = input("Enter service name: ")
                    data = self.get_password(service)
                    if data:
                        print(f"\\nService: {{service}}")
                        print(f"Username: {{data['username']}}")
                        print(f"Password: {{data['password']}}")
                        print(f"Created: {{data['created']}}")
                        print(f"Updated: {{data['updated']}}")
                
                elif choice == '3':
                    service = input("Enter service name: ")
                    if service in self.passwords:
                        current = self.passwords[service]
                        username = input(
                            f"Enter new username (current: {{current['username']}}, press Enter to keep): "
                        ) or None
                        password = getpass.getpass(
                            "Enter new password (press Enter to keep): "
                        ) or None
                        self.update_password(service, password, username)
                    else:
                        print("Service not found")
                
                elif choice == '4':
                    service = input("Enter service name to delete: ")
                    self.delete_password(service)
                
                elif choice == '5':
                    self.list_services()
                
                elif choice == '6':
                    print("Exiting...")
                    break
                
                else:
                    print("Invalid choice. Please try again.")
    
    if __name__ == '__main__':
        manager = PasswordManager()
        
        if not os.path.exists(manager.key_file):
            print("First-time setup. You need to initialize the password manager.")
            if manager.initialize():
                manager.interactive_menu()
        else:
            manager.interactive_menu()
    ''')

    with open(filename, 'w') as f:
        f.write(content)


def generate_program_8(dir_name):
    """Generate a stock market analysis program"""
    filename = os.path.join(dir_name, "stock_analysis.py")
    content = dedent(f'''\
    # Advanced Stock Market Analysis Program
    # Generated on {datetime.now()}
    # This program analyzes stock data with visualization and technical indicators
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import yfinance as yf
    from datetime import datetime, timedelta
    import mplfinance as mpf
    
    class StockAnalyzer:
        def __init__(self, ticker, period='1y'):
            self.ticker = ticker
            self.period = period
            self.data = None
            self.indicators = {}
        
        def fetch_data(self):
            """Download stock data from Yahoo Finance"""
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * int(self.period[:-1]))
            
            print(f"Fetching data for {{self.ticker}} ({{self.period}})...")
            self.data = yf.download(self.ticker, start=start_date, end=end_date)
            
            if self.data.empty:
                print("No data available for this ticker/period")
                return False
            
            self.data.index.name = 'Date'
            return True
        
        def calculate_sma(self, window=20):
            """Calculate Simple Moving Average"""
            self.indicators[f'SMA_{{window}}'] = self.data['Close'].rolling(window=window).mean()
        
        def calculate_ema(self, window=20):
            """Calculate Exponential Moving Average"""
            self.indicators[f'EMA_{{window}}'] = self.data['Close'].ewm(
                span=window, adjust=False
            ).mean()
        
        def calculate_rsi(self, window=14):
            """Calculate Relative Strength Index"""
            delta = self.data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            
            rs = avg_gain / avg_loss
            self.indicators[f'RSI_{{window}}'] = 100 - (100 / (1 + rs))
        
        def calculate_macd(self, fast=12, slow=26, signal=9):
            """Calculate MACD indicator"""
            ema_fast = self.data['Close'].ewm(span=fast, adjust=False).mean()
            ema_slow = self.data['Close'].ewm(span=slow, adjust=False).mean()
            
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            
            self.indicators['MACD'] = macd
            self.indicators['MACD_Signal'] = signal_line
            self.indicators['MACD_Hist'] = macd - signal_line
        
        def calculate_bollinger_bands(self, window=20, num_std=2):
            """Calculate Bollinger Bands"""
            sma = self.data['Close'].rolling(window=window).mean()
            std = self.data['Close'].rolling(window=window).std()
            
            self.indicators['BB_Upper'] = sma + (std * num_std)
            self.indicators['BB_Lower'] = sma - (std * num_std)
            self.indicators['BB_Middle'] = sma
        
        def plot_price_with_indicators(self):
            """Plot stock price with selected indicators"""
            if self.data is None:
                print("No data to plot")
                return
            
            plt.figure(figsize=(14, 8))
            
            # Plot closing price
            plt.plot(self.data.index, self.data['Close'], label='Close Price', color='blue', alpha=0.7)
            
            # Plot indicators
            for name, values in self.indicators.items():
                if 'SMA' in name or 'EMA' in name or 'BB' in name:
                    plt.plot(values.index, values, label=name, alpha=0.7)
            
            plt.title(f'{{self.ticker}} Stock Price with Indicators')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('stock_indicators.png')
            plt.close()
            print("Saved plot to stock_indicators.png")
        
        def plot_candlestick(self):
            """Plot candlestick chart with volume"""
            if self.data is None:
                print("No data to plot")
                return
            
            # Add indicators to the dataframe for mplfinance
            plot_data = self.data.copy()
            for name, values in self.indicators.items():
                if 'SMA' in name or 'EMA' in name or 'BB' in name:
                    plot_data[name] = values
            
            # Create style
            mc = mpf.make_marketcolors(
                up='g', down='r',
                wick={'up':'g', 'down':'r'},
                volume='in',
                edge='inherit'
            )
            style = mpf.make_mpf_style(marketcolors=mc, gridstyle=':')
            
            # Create plots
            plots = []
            for name in plot_data.columns:
                if 'SMA' in name or 'EMA' in name:
                    plots.append(mpf.make_addplot(plot_data[name]))
                elif 'BB' in name:
                    plots.append(mpf.make_addplot(plot_data[name], type='line'))
            
            # Plot
            mpf.plot(
                plot_data,
                type='candle',
                style=style,
                title=f'{{self.ticker}} Candlestick Chart',
                ylabel='Price',
                volume=True,
                addplot=plots,
                savefig='candlestick_chart.png',
                figratio=(12, 8),
                figscale=1.1
            )
            print("Saved candlestick chart to candlestick_chart.png")
        
        def generate_report(self):
            """Generate a text report with key statistics"""
            if self.data is None:
                print("No data for report")
                return
            
            report = f"Stock Analysis Report for {{self.ticker}}\\n"
            report += f"Date: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}\\n"
            report += f"Period: {{self.period}}\\n\\n"
            
            # Basic statistics
            latest = self.data.iloc[-1]
            report += "Latest Data:\\n"
            report += f"Date: {{latest.name.strftime('%Y-%m-%d')}}\\n"
            report += f"Open: {{latest['Open']:.2f}}\n"
            report += f"High: {{latest['High']:.2f}}\\n"
            report += f"Low: {{latest['Low']:.2f}}\\n"
            report += f"Close: {{latest['Close']:.2f}}\\n"
            report += f"Adj Close: {{latest['Adj Close']:.2f}}\\n"
            report += f"Volume: {{latest['Volume']:,}}\\n\\n"
            
            # Performance metrics
            start_price = self.data.iloc[0]['Close']
            end_price = self.data.iloc[-1]['Close']
            change = end_price - start_price
            pct_change = (change / start_price) * 100
            
            report += "Performance Metrics:\\n"
            report += f"Start Price: {{start_price:.2f}}\n"
            report += f"End Price: {{end_price:.2f}}\\n"
            report += f"Change: {{change:.2f}} ({{pct_change:.2f}}%)\\n\\n"
            
            # Volatility
            daily_returns = self.data['Close'].pct_change()
            volatility = daily_returns.std() * np.sqrt(252)  # Annualized
            
            report += "Risk Metrics:\\n"
            report += f"Annualized Volatility: {{volatility:.2%}}\\n\\n"
            
            # Save report
            with open('stock_report.txt', 'w') as f:
                f.write(report)
            print("Saved report to stock_report.txt")
    
    if __name__ == '__main__':
        # Example usage
        analyzer = StockAnalyzer('AAPL', '2y')
        
        if analyzer.fetch_data():
            # Calculate indicators
            analyzer.calculate_sma(50)
            analyzer.calculate_sma(200)
            analyzer.calculate_ema(20)
            analyzer.calculate_bollinger_bands()
            analyzer.calculate_rsi()
            analyzer.calculate_macd()
            
            # Generate outputs
            analyzer.plot_price_with_indicators()
            analyzer.plot_candlestick()
            analyzer.generate_report()
    ''')

    with open(filename, 'w') as f:
        f.write(content)


def generate_program_9(dir_name):
    """Generate a natural language processing program"""
    filename = os.path.join(dir_name, "nlp_processor.py")
    content = dedent(f'''\
    # Advanced Natural Language Processing Program
    # Generated on {datetime.now()}
    # This program demonstrates text processing with NLTK and spaCy
    
    import spacy
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk.probability import FreqDist
    from nltk.sentiment import SentimentIntensityAnalyzer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import string
    
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
    
    class NLPProcessor:
        def __init__(self, text=None, language='english'):
            self.text = text
            self.language = language
            self.nlp = spacy.load('en_core_web_sm')
            self.stop_words = set(stopwords.words(language))
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()
            self.sia = SentimentIntensityAnalyzer()
            
        def load_text(self, file_path):
            """Load text from a file"""
            with open(file_path, 'r', encoding='utf-8') as f:
                self.text = f.read()
            return self.text
        
        def preprocess_text(self, text=None):
            """Basic text preprocessing"""
            if text is None:
                text = self.text
            if not text:
                return ""
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            
            return text
        
        def tokenize_words(self, text=None):
            """Tokenize text into words"""
            if text is None:
                text = self.text
            return word_tokenize(text)
        
        def tokenize_sentences(self, text=None):
            """Tokenize text into sentences"""
            if text is None:
                text = self.text
            return sent_tokenize(text)
        
        def remove_stopwords(self, tokens):
            """Remove stopwords from token list"""
            return [word for word in tokens if word not in self.stop_words]
        
        def lemmatize_words(self, tokens):
            """Lemmatize words using WordNet"""
            return [self.lemmatizer.lemmatize(word) for word in tokens]
        
        def stem_words(self, tokens):
            """Stem words using Porter Stemmer"""
            return [self.stemmer.stem(word) for word in tokens]
        
        def get_word_frequencies(self, tokens, top_n=20):
            """Calculate word frequencies"""
            fdist = FreqDist(tokens)
            return fdist.most_common(top_n)
        
        def plot_word_frequencies(self, tokens, top_n=20):
            """Plot word frequency distribution"""
            fdist = FreqDist(tokens)
            fdist.plot(top_n, title='Word Frequency Distribution')
            plt.savefig('word_frequencies.png')
            plt.close()
            print("Saved word frequencies plot to word_frequencies.png")
        
        def generate_wordcloud(self, tokens, output_file='wordcloud.png'):
            """Generate a word cloud from tokens"""
            text = ' '.join(tokens)
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=200
            ).generate(text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.savefig(output_file)
            plt.close()
            print(f"Saved word cloud to {{output_file}}")
        
        def analyze_sentiment(self, text=None):
            """Perform sentiment analysis using VADER"""
            if text is None:
                text = self.text
            return self.sia.polarity_scores(text)
        
        def extract_entities(self, text=None):
            """Extract named entities using spaCy"""
            if text is None:
                text = self.text
            doc = self.nlp(text)
            return [(ent.text, ent.label_) for ent in doc.ents]
        
        def extract_noun_phrases(self, text=None):
            """Extract noun phrases using spaCy"""
            if text is None:
                text = self.text
            doc = self.nlp(text)
            return [chunk.text for chunk in doc.noun_chunks]
        
        def calculate_tfidf(self, documents):
            """Calculate TF-IDF scores for a list of documents"""
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top terms for each document
            results = []
            for i, doc in enumerate(documents):
                feature_index = tfidf_matrix[i,:].nonzero()[1]
                tfidf_scores = zip(
                    feature_index,
                    [tfidf_matrix[i, x] for x in feature_index]
                )
                sorted_scores = sorted(
                    tfidf_scores,
                    key=lambda x: x[1],
                    reverse=True
                )
                top_terms = [(feature_names[i], score) for (i, score) in sorted_scores[:10]]
                results.append({{
                    'document': doc[:50] + '...' if len(doc) > 50 else doc,
                    'top_terms': top_terms
                }})
            
            return results
        
        def full_pipeline(self, text=None):
            """Run full text processing pipeline"""
            if text is None:
                text = self.text
            
            print("\\n=== Text Processing Pipeline ===")
            
            # Preprocessing
            print("\\n1. Preprocessing text...")
            clean_text = self.preprocess_text(text)
            print(f"Sample: {{clean_text[:200]}}...")
            
            # Tokenization
            print("\\n2. Tokenizing words...")
            words = self.tokenize_words(clean_text)
            print(f"First 20 words: {{words[:20]}}")
            
            # Stopword removal
            print("\\n3. Removing stopwords...")
            filtered_words = self.remove_stopwords(words)
            print(f"First 20 filtered words: {{filtered_words[:20]}}")
            
            # Lemmatization
            print("\\n4. Lemmatizing words...")
            lemmatized = self.lemmatize_words(filtered_words)
            print(f"First 20 lemmatized words: {{lemmatized[:20]}}")
            
            # Word frequencies
            print("\\n5. Calculating word frequencies...")
            freq_dist = self.get_word_frequencies(lemmatized)
            print("Top 20 words:")
            for word, freq in freq_dist:
                print(f"{{word}}: {{freq}}")
            
            # Sentiment analysis
            print("\\n6. Analyzing sentiment...")
            sentiment = self.analyze_sentiment(text)
            print(f"Sentiment scores: {{sentiment}}")
            
            # Named entity recognition
            print("\\n7. Extracting named entities...")
            entities = self.extract_entities(text)
            print("Named entities found:")
            for entity, label in entities[:10]:  # Show first 10
                print(f"{{entity}} ({{label}})")
            
            # Generate visualizations
            print("\\n8. Generating visualizations...")
            self.plot_word_frequencies(lemmatized)
            self.generate_wordcloud(lemmatized)
            
            return {
                'clean_text': clean_text,
                'words': words,
                'filtered_words': filtered_words,
                'lemmatized': lemmatized,
                'frequencies': freq_dist,
                'sentiment': sentiment,
                'entities': entities
            }
    
    if __name__ == '__main__':
        # Example text (could be loaded from file)
        sample_text = """
        Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language."""
        
        processor = NLPProcessor(sample_text)
        
        # Run full pipeline
        results = processor.full_pipeline()
        
        # Additional analysis with multiple documents
        documents = [
            "Machine learning is the study of computer algorithms that improve automatically through experience.",
            "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
            "Natural language processing enables computers to understand, interpret, and manipulate human language."
        ]
        
        print("\\n=== TF-IDF Analysis ===")
        tfidf_results = processor.calculate_tfidf(documents)
        for result in tfidf_results:
            print(f"\\nDocument: {{result['document']}}")
            print("Top terms:")
            for term, score in result['top_terms']:
                print(f"{{term}}: {{score:.4f}}")
    ''')

    with open(filename, 'w') as f:
        f.write(content)


def generate_program_10(dir_name):
    """Generate a computer vision program with OpenCV"""
    filename = os.path.join(dir_name, "computer_vision.py")
    content = dedent(f'''\
    # Advanced Computer Vision Program
    # Generated on {datetime.now()}
    # This program demonstrates various computer vision techniques using OpenCV
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    
    class VisionProcessor:
        def __init__(self, image_path=None):
            self.image_path = image_path
            self.image = None
            self.gray_image = None
            self.hsv_image = None
            self.keypoints = None
            self.descriptors = None
            
            if image_path:
                self.load_image(image_path)
        
        def load_image(self, image_path):
            """Load an image from file"""
            self.image_path = image_path
            self.image = cv2.imread(image_path)
            if self.image is None:
                raise ValueError(f"Could not load image from {{image_path}}")
            
            # Convert color spaces
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            
            return self.image
        
        def display_image(self, image=None, title='Image', cmap=None):
            """Display an image using matplotlib"""
            if image is None:
                image = self.image
            
            if len(image.shape) == 2:  # Grayscale
                plt.imshow(image, cmap='gray')
            else:
                # Convert BGR to RGB for matplotlib
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.imshow(image_rgb, cmap=cmap)
            
            plt.title(title)
            plt.axis('off')
            plt.show()
        
        def save_image(self, image, filename):
            """Save an image to file"""
            cv2.imwrite(filename, image)
            print(f"Image saved as {{filename}}")
        
        def resize_image(self, width=None, height=None, inter=cv2.INTER_AREA):
            """Resize image while maintaining aspect ratio"""
            if width is None and height is None:
                return self.image
                
            h, w = self.image.shape[:2]
            
            if width is None:
                r = height / float(h)
                dim = (int(w * r), height)
            else:
                r = width / float(w)
                dim = (width, int(h * r))
            
            resized = cv2.resize(self.image, dim, interpolation=inter)
            return resized
        
        def detect_edges(self, method='canny', low_thresh=50, high_thresh=150):
            """Detect edges using various methods"""
            if method == 'canny':
                edges = cv2.Canny(self.gray_image, low_thresh, high_thresh)
            elif method == 'sobel':
                sobelx = cv2.Sobel(self.gray_image, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(self.gray_image, cv2.CV_64F, 0, 1, ksize=3)
                edges = np.sqrt(sobelx**2 + sobely**2)
                edges = np.uint8(edges * 255 / np.max(edges))
            elif method == 'laplacian':
                edges = cv2.Laplacian(self.gray_image, cv2.CV_64F)
                edges = np.uint8(np.absolute(edges))
            else:
                raise ValueError("Invalid edge detection method")
            
            return edges
        
        def detect_corners(self, max_corners=25, quality=0.01, min_dist=10):
            """Detect corners using Shi-Tomasi method"""
            corners = cv2.goodFeaturesToTrack(
                self.gray_image,
                max_corners,
                quality,
                min_dist
            )
            
            corner_image = self.image.copy()
            if corners is not None:
                corners = np.int0(corners)
                for corner in corners:
                    x, y = corner.ravel()
                    cv2.circle(corner_image, (x, y), 3, (0, 255, 0), -1)
            
            return corner_image
        
        def detect_features(self, method='orb', n_features=500):
            """Detect and compute keypoints and descriptors"""
            if method == 'orb':
                detector = cv2.ORB_create(n_features)
            elif method == 'sift':
                detector = cv2.SIFT_create(n_features)
            elif method == 'surf':
                detector = cv2.xfeatures2d.SURF_create(n_features)
            elif method == 'akaze':
                detector = cv2.AKAZE_create()
            else:
                raise ValueError("Invalid feature detection method")
            
            self.keypoints, self.descriptors = detector.detectAndCompute(
                self.gray_image, None
            )
            
            return self.keypoints, self.descriptors
        
        def draw_keypoints(self, color=(0, 255, 0)):
            """Draw detected keypoints on the image"""
            if self.keypoints is None:
                raise ValueError("No keypoints detected. Run detect_features first.")
            
            keypoint_image = cv2.drawKeypoints(
                self.image,
                self.keypoints,
                None,
                color=color,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            
            return keypoint_image
        
        def match_features(self, other_processor, method='bf', draw_matches=False):
            """Match features between two images"""
            if self.descriptors is None or other_processor.descriptors is None:
                raise ValueError("Descriptors not computed for one or both images")
            
            if method == 'bf':
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            elif method == 'flann':
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
                search_params = dict(checks=50)
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
            else:
                raise ValueError("Invalid matching method")
            
            matches = matcher.match(self.descriptors, other_processor.descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if draw_matches:
                match_image = cv2.drawMatches(
                    self.image, self.keypoints,
                    other_processor.image, other_processor.keypoints,
                    matches[:50], None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                return matches, match_image
            
            return matches
        
        def detect_faces(self, scale_factor=1.1, min_neighbors=5):
            """Detect faces using Haar cascades"""
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            faces = face_cascade.detectMultiScale(
                self.gray_image,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(30, 30)
            )
            
            face_image = self.image.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(face_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            return faces, face_image
        
        def detect_colors(self, n_colors=5):
            """Detect dominant colors using k-means clustering"""
            pixels = self.image.reshape(-1, 3)
            pixels = np.float32(pixels)
            
            # Perform k-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
            _, labels, centers = cv2.kmeans(
                pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )
            
            # Convert centers to uint8
            centers = np.uint8(centers)
            
            # Calculate percentages
            unique, counts = np.unique(labels, return_counts=True)
            percentages = counts / counts.sum()
            
            # Sort by percentage
            sorted_indices = np.argsort(percentages)[::-1]
            dominant_colors = centers[unique[sorted_indices]]
            color_percentages = percentages[sorted_indices]
            
            return dominant_colors, color_percentages
        
        def plot_color_palette(self, colors, percentages):
            """Plot a color palette with percentages"""
            plt.figure(figsize=(8, 2))
            for i, (color, percent) in enumerate(zip(colors, percentages)):
                plt.fill_between([i, i+1], 0, 1, color=color/255)
                plt.text(i+0.5, 0.5, f"{percent:.1%}",
                         ha='center', va='center', color='white')
            plt.xlim(0, len(colors))
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('color_palette.png')
            plt.close()
            print("Saved color palette to color_palette.png")
        
        def apply_filter(self, filter_type, kernel_size=3):
            """Apply various filters to the image"""
            if filter_type == 'blur':
                return cv2.blur(self.image, (kernel_size, kernel_size))
            elif filter_type == 'gaussian':
                return cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
            elif filter_type == 'median':
                return cv2.medianBlur(self.image, kernel_size)
            elif filter_type == 'bilateral':
                return cv2.bilateralFilter(self.image, kernel_size, 75, 75)
            elif filter_type == 'sharpen':
                kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
                return cv2.filter2D(self.image, -1, kernel)
            else:
                raise ValueError("Invalid filter type")
        
        def apply_threshold(self, method='otsu'):
            """Apply thresholding to the grayscale image"""
            if method == 'binary':
                _, thresh = cv2.threshold(
                    self.gray_image, 127, 255, cv2.THRESH_BINARY
                )
            elif method == 'otsu':
                _, thresh = cv2.threshold(
                    self.gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
            elif method == 'adaptive':
                thresh = cv2.adaptiveThreshold(
                    self.gray_image, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
            else:
                raise ValueError("Invalid threshold method")
            
            return thresh
        
        def detect_contours(self, mode='external', method='simple'):
            """Find contours in a binary image"""
            thresh = self.apply_threshold('otsu')
            
            if mode == 'external':
                mode = cv2.RETR_EXTERNAL
            elif mode == 'all':
                mode = cv2.RETR_LIST
            elif mode == 'tree':
                mode = cv2.RETR_TREE
            else:
                raise ValueError("Invalid contour mode")
                
            if method == 'simple':
                method = cv2.CHAIN_APPROX_SIMPLE
            elif method == 'none':
                method = cv2.CHAIN_APPROX_NONE
            else:
                raise ValueError("Invalid contour method")
            
            contours, _ = cv2.findContours(thresh, mode, method)
            
            contour_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
            
            return contours, contour_image
        
        def process_pipeline(self, output_dir='output'):
            """Run a complete computer vision pipeline"""
            import os
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            print("\\n=== Computer Vision Pipeline ===")
            
            # 1. Display original image
            print("1. Displaying original image...")
            self.display_image(title='Original Image')
            
            # 2. Edge detection
            print("2. Performing edge detection...")
            edges = self.detect_edges()
            self.save_image(edges, os.path.join(output_dir, 'edges.jpg'))
            
            # 3. Corner detection
            print("3. Detecting corners...")
            corners = self.detect_corners()
            self.save_image(corners, os.path.join(output_dir, 'corners.jpg'))
            
            # 4. Feature detection
            print("4. Detecting features...")
            self.detect_features()
            keypoints = self.draw_keypoints()
            self.save_image(keypoints, os.path.join(output_dir, 'keypoints.jpg'))
            
            # 5. Face detection
            print("5. Detecting faces...")
            _, faces = self.detect_faces()
            self.save_image(faces, os.path.join(output_dir, 'faces.jpg'))
            
            # 6. Color analysis
            print("6. Analyzing colors...")
            colors, percentages = self.detect_colors()
            self.plot_color_palette(colors, percentages)
            
            # 7. Thresholding
            print("7. Applying thresholding...")
            thresh = self.apply_threshold()
            self.save_image(thresh, os.path.join(output_dir, 'threshold.jpg'))
            
            # 8. Contour detection
            print("8. Finding contours...")
            _, contours = self.detect_contours()
            self.save_image(contours, os.path.join(output_dir, 'contours.jpg'))
            
            print("\\nPipeline completed. Results saved to", output_dir)
    
    if __name__ == '__main__':
        pass
        processor = VisionProcessor('example.jpg')  # Replace with your image path
        
        # Run full pipeline
        processor.process_pipeline()
        
        # Or use individual methods
        # edges = processor.detect_edges()
        # processor.display_image(edges, title='Edges')
        
        # colors, percentages = processor.detect_colors()
        # processor.plot_color_palette(colors, percentages)
    ''')

    with open(filename, 'w') as f:
        f.write(content)


def generate_program_11(dir_name):
    """Generate a blockchain simulation program"""
    filename = os.path.join(dir_name, "blockchain.py")
    content = dedent(f'''\
    # Advanced Blockchain Simulation
    # Generated on {datetime.now()}
    # This program implements a basic blockchain with proof-of-work and transactions
    
    import hashlib
    import json
    from time import time
    from uuid import uuid4
    from typing import List, Dict
    import random
    
    class Transaction:
        def __init__(self, sender: str, recipient: str, amount: float):
            self.sender = sender
            self.recipient = recipient
            self.amount = amount
            self.id = str(uuid4()).replace('-', '')
            self.timestamp = time()
        
        def to_dict(self) -> Dict:
            return {
                'id': self.id,
                'sender': self.sender,
                'recipient': self.recipient,
                'amount': self.amount,
                'timestamp': self.timestamp
            }
        
        def __repr__(self) -> str:
            return f"Transaction({{self.sender}} -> {{self.recipient}}: {{self.amount}})"
    
    class Block:
        def __init__(self, index: int, transactions: List[Transaction], 
                     previous_hash: str, nonce: int = 0, timestamp: float = None):
            self.index = index
            self.transactions = transactions
            self.previous_hash = previous_hash
            self.nonce = nonce
            self.timestamp = timestamp or time()
            self.hash = self.calculate_hash()
        
        def calculate_hash(self) -> str:
            block_string = json.dumps({
                'index': self.index,
                'transactions': [tx.to_dict() for tx in self.transactions],
                'previous_hash': self.previous_hash,
                'nonce': self.nonce,
                'timestamp': self.timestamp
            }, sort_keys=True).encode()
            
            return hashlib.sha256(block_string).hexdigest()
        
        def mine_block(self, difficulty: int) -> None:
            target = '0' * difficulty
            while self.hash[:difficulty] != target:
                self.nonce += 1
                self.hash = self.calculate_hash()
        
        def __repr__(self) -> str:
            return f"Block({{self.index}}, Hash: {{self.hash[:10]}}..., Prev: {{self.previous_hash[:10]}}..., Transactions: {{len(self.transactions)}})"
    
    class Blockchain:
        def __init__(self):
            self.chain: List[Block] = []
            self.pending_transactions: List[Transaction] = []
            self.difficulty = 4
            self.mining_reward = 10
            self.address_balances = {}
            
            # Create genesis block
            self.create_genesis_block()
        
        def create_genesis_block(self) -> None:
            genesis_block = Block(0, [], '0')
            genesis_block.mine_block(self.difficulty)
            self.chain.append(genesis_block)
        
        def get_last_block(self) -> Block:
            return self.chain[-1]
        
        def mine_pending_transactions(self, mining_reward_address: str) -> None:
            if not self.pending_transactions:
                print("No transactions to mine")
                return
            
            # Add mining reward transaction
            reward_tx = Transaction(
                "network",
                mining_reward_address,
                self.mining_reward
            )
            self.pending_transactions.append(reward_tx)
            
            # Create new block with pending transactions
            new_block = Block(
                len(self.chain),
                self.pending_transactions,
                self.get_last_block().hash
            )
            
            print(f"Mining block {{new_block.index}}...")
            new_block.mine_block(self.difficulty)
            
            # Add block to chain
            self.chain.append(new_block)
            
            # Update balances
            for tx in new_block.transactions:
                self.update_balances(tx)
            
            # Reset pending transactions
            self.pending_transactions = []
            
            print(f"Block mined successfully: {{new_block.hash}}")
        
        def add_transaction(self, transaction: Transaction) -> None:
            if not transaction.sender or not transaction.recipient:
                raise ValueError("Transaction must include sender and recipient")
            
            if transaction.amount <= 0:
                raise ValueError("Transaction amount must be positive")
            
            # Verify sender has sufficient balance
            sender_balance = self.get_balance(transaction.sender)
            if sender_balance < transaction.amount and transaction.sender != "network":
                raise ValueError("Insufficient balance")
            
            self.pending_transactions.append(transaction)
            print(f"Transaction added: {{transaction}}")
        
        def get_balance(self, address: str) -> float:
            if address not in self.address_balances:
                return 0.0
            
            return self.address_balances[address]
        
        def update_balances(self, transaction: Transaction) -> None:
            # Skip network transactions (mining rewards)
            if transaction.sender != "network":
                if transaction.sender not in self.address_balances:
                    self.address_balances[transaction.sender] = 0
                self.address_balances[transaction.sender] -= transaction.amount
            
            if transaction.recipient not in self.address_balances:
                self.address_balances[transaction.recipient] = 0
            self.address_balances[transaction.recipient] += transaction.amount
        
        def is_chain_valid(self) -> bool:
            for i in range(1, len(self.chain)):
                current_block = self.chain[i]
                previous_block = self.chain[i-1]
                
                # Check block hash integrity
                if current_block.hash != current_block.calculate_hash():
                    print(f"Block {{current_block.index}} hash is invalid")
                    return False
                
                # Check chain linkage
                if current_block.previous_hash != previous_block.hash:
                    print(f"Block {{current_block.index}} previous hash doesn't match")
                    return False
                
                # Check proof of work
                if current_block.hash[:self.difficulty] != '0' * self.difficulty:
                    print(f"Block {{current_block.index}} doesn't meet difficulty requirement")
                    return False
            
            return True
        
        def __repr__(self) -> str:
            return f"Blockchain(Blocks: {{len(self.chain)}}, Pending TXs: {{len(self.pending_transactions)}})"
    
    class Wallet:
        def __init__(self, blockchain: Blockchain):
            self.blockchain = blockchain
            self.address = str(uuid4()).replace('-', '')
            print(f"Created new wallet with address: {{self.address}}")
        
        def send_money(self, recipient: str, amount: float) -> None:
            if amount <= 0:
                print("Amount must be positive")
                return
            
            if self.address == recipient:
                print("Cannot send money to yourself")
                return
            
            balance = self.blockchain.get_balance(self.address)
            if balance < amount:
                print(f"Insufficient balance ({{balance}} available)")
                return
            
            transaction = Transaction(self.address, recipient, amount)
            self.blockchain.add_transaction(transaction)
        
        def get_balance(self) -> float:
            return self.blockchain.get_balance(self.address)
    
    def simulate_blockchain():
        print("=== Blockchain Simulation ===")
        
        # Create blockchain
        blockchain = Blockchain()
        
        # Create some wallets
        wallets = [Wallet(blockchain) for _ in range(3)]
        miner = wallets[0]
        user1 = wallets[1]
        user2 = wallets[2]
        
        # Mine some blocks to get initial coins
        print("\\nMining initial blocks...")
        for _ in range(3):
            blockchain.mine_pending_transactions(miner.address)
        
        print(f"\\nMiner balance: {{miner.get_balance()}}")
        
        # Simulate some transactions
        print("\\nSimulating transactions...")
        miner.send_money(user1.address, 15)
        miner.send_money(user2.address, 10)
        user1.send_money(user2.address, 5)
        user2.send_money(user1.address, 3)
        
        # Mine the transactions
        print("\\nMining a block with transactions...")
        blockchain.mine_pending_transactions(miner.address)
        
        # Print balances
        print("\\nFinal balances:")
        for i, wallet in enumerate(wallets):
            print(f"Wallet {{i+1}} ({{wallet.address[:8]}}...): {{wallet.get_balance()}}")
        
        # Validate chain
        print("\\nValidating blockchain...")
        is_valid = blockchain.is_chain_valid()
        print(f"Blockchain is valid: {{is_valid}}")
        
        # Print chain info
        print("\\nBlockchain info:")
        print(blockchain)
        for block in blockchain.chain:
            print(f"- {{block}}")
    
    if __name__ == '__main__':
        simulate_blockchain()
    ''')

    with open(filename, 'w') as f:
        f.write(content)


def generate_program_12(dir_name):
    """Generate a genetic algorithm implementation"""
    filename = os.path.join(dir_name, "genetic_algorithm.py")
    content = dedent(f'''\
    # Advanced Genetic Algorithm Implementation
    # Generated on {datetime.now()}
    # This program demonstrates a genetic algorithm for optimization problems
    
    import random
    import numpy as np
    from typing import List, Tuple, Callable
    from functools import partial
    import matplotlib.pyplot as plt
    
    class GeneticAlgorithm:
        def __init__(self, 
                     fitness_func: Callable[[List[float]], float],
                     population_size: int = 100,
                     gene_count: int = 10,
                     gene_range: Tuple[float, float] = (0, 1),
                     crossover_rate: float = 0.8,
                     mutation_rate: float = 0.1,
                     mutation_scale: float = 0.5,
                     elitism_ratio: float = 0.1,
                     selection_pressure: float = 1.5):
            """
            Initialize the genetic algorithm with parameters
            
            Args:
                fitness_func: Function that evaluates a chromosome and returns a fitness score
                population_size: Number of individuals in each generation
                gene_count: Number of genes in each chromosome
                gene_range: Minimum and maximum value for each gene
                crossover_rate: Probability of crossover between parents (0-1)
                mutation_rate: Probability of mutation for each gene (0-1)
                mutation_scale: Scale factor for Gaussian mutation
                elitism_ratio: Fraction of top individuals to carry over to next generation
                selection_pressure: Pressure for selecting fitter individuals (1 = no pressure)
            """
            self.fitness_func = fitness_func
            self.population_size = population_size
            self.gene_count = gene_count
            self.gene_range = gene_range
            self.crossover_rate = crossover_rate
            self.mutation_rate = mutation_rate
            self.mutation_scale = mutation_scale
            self.elitism_ratio = elitism_ratio
            self.selection_pressure = selection_pressure
            
            # Validate parameters
            if not (0 <= crossover_rate <= 1):
                raise ValueError("Crossover rate must be between 0 and 1")
            if not (0 <= mutation_rate <= 1):
                raise ValueError("Mutation rate must be between 0 and 1")
            if elitism_ratio < 0 or elitism_ratio > 0.5:
                raise ValueError("Elitism ratio should be between 0 and 0.5")
            if selection_pressure < 1:
                raise ValueError("Selection pressure should be >= 1")
            
            # Initialize population
            self.population = self.initialize_population()
            self.best_chromosome = None
            self.best_fitness = -np.inf
            self.fitness_history = []
            self.avg_fitness_history = []
        
        def initialize_population(self) -> List[List[float]]:
            """Create initial random population"""
            return [
                [
                    random.uniform(self.gene_range[0], self.gene_range[1])
                    for _ in range(self.gene_count)
                ]
                for _ in range(self.population_size)
            ]
        
        def evaluate_population(self) -> List[float]:
            """Evaluate fitness of all individuals in population"""
            fitness_scores = [self.fitness_func(ind) for ind in self.population]
            
            # Update best solution
            max_fitness = max(fitness_scores)
            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness
                self.best_chromosome = self.population[fitness_scores.index(max_fitness)].copy()
            
            # Record statistics
            self.fitness_history.append(self.best_fitness)
            self.avg_fitness_history.append(np.mean(fitness_scores))
            
            return fitness_scores
        
        def select_parents(self, fitness_scores: List[float]) -> List[List[float]]:
            """Select parents for next generation using rank-based selection"""
            # Rank individuals (1 is best)
            ranked = sorted(zip(self.population, fitness_scores), 
                          key=lambda x: x[1], reverse=True)
            population, _ = zip(*ranked)
            population = list(population)
            
            # Calculate selection probabilities
            ranks = np.arange(1, len(population) + 1)
            probabilities = (2 - self.selection_pressure) / len(population) + (2 * (self.selection_pressure - 1) * (len(population) - ranks + 1)) / \
                           (len(population) * (len(population) - 1))
            
            # Select parents
            parents = random.choices(
                population,
                weights=probabilities,
                k=self.population_size - int(self.elitism_ratio * self.population_size)
            )
            
            return parents
        
        def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
            """Perform crossover between two parents to produce two children"""
            if random.random() > self.crossover_rate:
                return parent1.copy(), parent2.copy()
            
            # Blend crossover (BLX-α)
            alpha = 0.5
            child1, child2 = [], []
            
            for g1, g2 in zip(parent1, parent2):
                d = abs(g1 - g2)
                min_g = min(g1, g2) - alpha * d
                max_g = max(g1, g2) + alpha * d
                
                # Ensure new genes are within bounds
                min_g = max(min_g, self.gene_range[0])
                max_g = min(max_g, self.gene_range[1])
                
                child1.append(random.uniform(min_g, max_g))
                child2.append(random.uniform(min_g, max_g))
            
            return child1, child2
        
        def mutate(self, chromosome: List[float]) -> List[float]:
            """Apply mutation to a chromosome"""
            mutated = chromosome.copy()
            
            for i in range(len(mutated)):
                if random.random() < self.mutation_rate:
                    # Gaussian mutation
                    mutated[i] += random.gauss(0, self.mutation_scale)
                    # Ensure gene stays within bounds
                    mutated[i] = np.clip(mutated[i], *self.gene_range)
            
            return mutated
        
        def create_next_generation(self, fitness_scores: List[float]) -> None:
            """Create next generation through selection, crossover and mutation"""
            # Select parents
            parents = self.select_parents(fitness_scores)
            
            # Perform elitism - carry over top individuals unchanged
            elite_count = int(self.elitism_ratio * self.population_size)
            ranked = sorted(zip(self.population, fitness_scores), 
                          key=lambda x: x[1], reverse=True)
            elite = [ind for ind, _ in ranked[:elite_count]]
            
            # Create children through crossover
            children = []
            for i in range(0, len(parents), 2):
                if i+1 >= len(parents):
                    break  # Skip if odd number of parents
                
                child1, child2 = self.crossover(parents[i], parents[i+1])
                children.extend([child1, child2])
            
            # Apply mutation
            mutated_children = [self.mutate(child) for child in children]
            
            # Combine elite and children to form new population
            self.population = elite + mutated_children
            
            # If population size is odd, add one more random individual
            if len(self.population) < self.population_size:
                self.population.append(self.initialize_population()[0])
        
        def run(self, generations: int = 100) -> Tuple[List[float], float]:
            """Run the genetic algorithm for specified number of generations"""
            for gen in range(generations):
                fitness_scores = self.evaluate_population()
                self.create_next_generation(fitness_scores)
                
                # Print progress
                if (gen + 1) % 10 == 0:
                    print(f"Generation {{gen+1}}: Best = {{self.best_fitness:.4f}}, "f"Avg = {{self.avg_fitness_history[-1]:.4f}}")
            
            return self.best_chromosome, self.best_fitness
        
        def plot_progress(self) -> None:
            """Plot the fitness progress over generations"""
            plt.figure(figsize=(10, 6))
            plt.plot(self.fitness_history, label='Best Fitness')
            plt.plot(self.avg_fitness_history, label='Average Fitness')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('Genetic Algorithm Progress')
            plt.legend()
            plt.grid(True)
            plt.savefig('ga_progress.png')
            plt.close()
            print("Saved progress plot to ga_progress.png")
    
    def example_fitness_function(chromosome: List[float]) -> float:
        """Example fitness function: Rastrigin function (minimization problem)"""
        A = 10
        n = len(chromosome)
        return - (A * n + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in chromosome]))
    
    def example_run():
        print("=== Genetic Algorithm Example ===")
        print("Optimizing Rastrigin function (10 dimensions)")
        
        # Initialize GA
        ga = GeneticAlgorithm(
            fitness_func=example_fitness_function,
            population_size=100,
            gene_count=10,
            gene_range=(-5.12, 5.12),
            crossover_rate=0.9,
            mutation_rate=0.1,
            mutation_scale=0.5,
            elitism_ratio=0.1,
            selection_pressure=1.7
        )
        
        # Run optimization
        best_solution, best_fitness = ga.run(generations=200)
        
        # Display results
        print("\\n=== Results ===")
        print(f"Best solution found: {{best_solution}}")
        print(f"Best fitness value: {{best_fitness:.4f}}")
        
        # Plot progress
        ga.plot_progress()
    
    if __name__ == '__main__':
        example_run()
    ''')

    with open(filename, 'w') as f:
        f.write(content)


def generate_program_13(dir_name):
    """Generate a neural network from scratch"""
    filename = os.path.join(dir_name, "neural_network.py")
    content = dedent(f'''\
    # Advanced Neural Network from Scratch
    # Generated on {datetime.now()}
    # This program implements a neural network with backpropagation without ML libraries
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    class NeuralNetwork:
        def __init__(self, layer_sizes, learning_rate=0.01, reg_lambda=0.01):
            """
            Initialize the neural network
            
            Args:
                layer_sizes: List of layer sizes (input, hidden..., output)
                learning_rate: Learning rate for gradient descent
                reg_lambda: Regularization strength
            """
            self.layer_sizes = layer_sizes
            self.learning_rate = learning_rate
            self.reg_lambda = reg_lambda
            self.weights = []
            self.biases = []
            self.loss_history = []
            self.val_loss_history = []
            
            # Initialize weights and biases
            for i in range(len(layer_sizes) - 1):
                # He initialization
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
                b = np.zeros((1, layer_sizes[i+1]))
                self.weights.append(w)
                self.biases.append(b)
        
        def sigmoid(self, z):
            """Sigmoid activation function"""
            return 1 / (1 + np.exp(-z))
        
        def sigmoid_derivative(self, z):
            """Derivative of sigmoid function"""
            s = self.sigmoid(z)
            return s * (1 - s)
        
        def relu(self, z):
            """ReLU activation function"""
            return np.maximum(0, z)
        
        def relu_derivative(self, z):
            """Derivative of ReLU function"""
            return (z > 0).astype(float)
        
        def softmax(self, z):
            """Softmax activation function"""
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        def forward_propagate(self, X):
            """Perform forward propagation through the network"""
            activations = [X]
            zs = []
            a = X
            
            # Hidden layers (use ReLU)
            for i in range(len(self.weights) - 1):
                z = np.dot(a, self.weights[i]) + self.biases[i]
                a = self.relu(z)
                zs.append(z)
                activations.append(a)
            
            # Output layer (use softmax)
            z = np.dot(a, self.weights[-1]) + self.biases[-1]
            a = self.softmax(z)
            zs.append(z)
            activations.append(a)
            
            return activations, zs
        
        def compute_loss(self, y, y_hat):
            """Compute cross-entropy loss with L2 regularization"""
            m = y.shape[0]
            corect_logprobs = -np.log(y_hat[range(m), y.argmax(axis=1)])
            data_loss = np.sum(corect_logprobs) / m
            
            # Add L2 regularization
            reg_loss = 0
            for w in self.weights:
                reg_loss += 0.5 * self.reg_lambda * np.sum(w * w)
            
            return data_loss + reg_loss
        
        def backward_propagate(self, X, y, activations, zs):
            """Perform backward propagation to compute gradients"""
            m = X.shape[0]
            deltas = [None] * len(self.weights)
            
            # Output layer gradient
            y_hat = activations[-1]
            delta = y_hat - y
            deltas[-1] = delta
            
            # Hidden layers gradient
            for l in range(len(deltas) - 2, -1, -1):
                delta = np.dot(deltas[l+1], self.weights[l+1].T) * self.relu_derivative(zs[l])
                deltas[l] = delta
            
            # Compute gradients
            grads_w = []
            grads_b = []
            for l in range(len(deltas)):
                grad_w = np.dot(activations[l].T, deltas[l]) / m
                grad_w += self.reg_lambda * self.weights[l]  # L2 regularization
                grad_b = np.sum(deltas[l], axis=0, keepdims=True) / m
                
                grads_w.append(grad_w)
                grads_b.append(grad_b)
            
            return grads_w, grads_b
        
        def update_parameters(self, grads_w, grads_b):
            """Update weights and biases using gradients"""
            for l in range(len(self.weights)):
                self.weights[l] -= self.learning_rate * grads_w[l]
                self.biases[l] -= self.learning_rate * grads_b[l]
        
        def train(self, X, y, X_val=None, y_val=None, epochs=1000, batch_size=32, verbose=100):
            """Train the neural network"""
            m = X.shape[0]
            
            for epoch in range(epochs):
                # Mini-batch gradient descent
                indices = np.random.permutation(m)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                
                for i in range(0, m, batch_size):
                    X_batch = X_shuffled[i:i+batch_size]
                    y_batch = y_shuffled[i:i+batch_size]
                    
                    # Forward and backward pass
                    activations, zs = self.forward_propagate(X_batch)
                    grads_w, grads_b = self.backward_propagate(X_batch, y_batch, activations, zs)
                    self.update_parameters(grads_w, grads_b)
                
                # Compute loss
                activations, _ = self.forward_propagate(X)
                loss = self.compute_loss(y, activations[-1])
                self.loss_history.append(loss)
                
                # Compute validation loss if validation data provided
                if X_val is not None and y_val is not None:
                    val_activations, _ = self.forward_propagate(X_val)
                    val_loss = self.compute_loss(y_val, val_activations[-1])
                    self.val_loss_history.append(val_loss)
                
                # Print progress
                if verbose and (epoch + 1) % verbose == 0:
                    msg = f"Epoch {{epoch+1}}/{{epochs}} - loss: {{loss:.4f}}"
                    if X_val is not None and y_val is not None:
                        msg += f" - val_loss: {{val_loss:.4f}}"
                    print(msg)
        
        def predict(self, X):
            """Make predictions using the trained network"""
            activations, _ = self.forward_propagate(X)
            return activations[-1].argmax(axis=1)
        
        def plot_loss(self):
            """Plot training and validation loss history"""
            plt.figure(figsize=(10, 6))
            plt.plot(self.loss_history, label='Training Loss')
            if self.val_loss_history:
                plt.plot(self.val_loss_history, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True)
            plt.savefig('training_loss.png')
            plt.close()
            print("Saved training plot to training_loss.png")
    
    def example_run():
        print("=== Neural Network Example ===")
        print("Training on moons dataset with 2 hidden layers")
        
        # Generate and prepare data
        X, y = make_moons(n_samples=2000, noise=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Convert to one-hot encoding
        y_train_onehot = np.zeros((y_train.shape[0], 2))
        y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1
        
        y_test_onehot = np.zeros((y_test.shape[0], 2))
        y_test_onehot[np.arange(y_test.shape[0]), y_test] = 1
        
        # Initialize and train network
        nn = NeuralNetwork(
            layer_sizes=[2, 64, 64, 2],  # 2 input, 2 hidden (64 neurons each), 2 output
            learning_rate=0.01,
            reg_lambda=0.001
        )
        
        nn.train(
            X_train, y_train_onehot,
            X_val=X_test, y_val=y_test_onehot,
            epochs=500,
            batch_size=32,
            verbose=50
        )
        
        # Evaluate
        y_pred = nn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\\nTest accuracy: {{accuracy:.4f}}")
        
        # Plot decision boundary
        def plot_decision_boundary(pred_func, X, y):
            # Set min and max values
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            h = 0.01
            
            # Generate grid of points
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            # Predict for each point
            Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot contour and training examples
            plt.figure(figsize=(10, 6))
            plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
            plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='k')
            plt.title("Decision Boundary")
            plt.savefig('decision_boundary.png')
            plt.close()
            print("Saved decision boundary plot to decision_boundary.png")
        
        plot_decision_boundary(
            lambda x: nn.predict(x),
            X_test,
            y_test
        )
        
        # Plot training progress
        nn.plot_loss()
    
    if __name__ == '__main__':
        example_run()
    ''')

    with open(filename, 'w') as f:
        f.write(content)


def generate_program_14(dir_name):
    """Generate a reinforcement learning program"""
    filename = os.path.join(dir_name, "reinforcement_learning.py")
    content = dedent(f'''\
    # Advanced Reinforcement Learning Program
    # Generated on {datetime.now()}
    # This program implements Q-learning and Deep Q-Network for the CartPole environment
    
    import numpy as np
    import random
    from collections import deque
    import gym
    from gym import wrappers
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    import matplotlib.pyplot as plt
    
    class QLearningAgent:
        def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
            self.env = env
            self.alpha = alpha  # Learning rate
            self.gamma = gamma  # Discount factor
            self.epsilon = epsilon  # Exploration rate
            self.epsilon_decay = epsilon_decay
            self.epsilon_min = epsilon_min
            self.q_table = {}  # State-action value table
            self.rewards = []
        
        def discretize_state(self, state):
            """Convert continuous state to discrete bins"""
            # CartPole state: [position, velocity, angle, angular velocity]
            discretized = (
                int(np.digitize(state[0], np.linspace(-2.4, 2.4, 10))),
                int(np.digitize(state[1], np.linspace(-3.0, 3.0, 10))),
                int(np.digitize(state[2], np.linspace(-0.5, 0.5, 10))),
                int(np.digitize(state[3], np.linspace(-2.0, 2.0, 10)))
            )
            return discretized
        
        def get_q_value(self, state, action):
            """Get Q-value for state-action pair, initialize if not exists"""
            if state not in self.q_table:
                self.q_table[state] = np.zeros(self.env.action_space.n)
            return self.q_table[state][action]
        
        def choose_action(self, state):
            """Epsilon-greedy action selection"""
            if random.random() < self.epsilon:
                return self.env.action_space.sample()  # Random action
            else:
                return np.argmax(self.q_table[state])  # Best known action
        
        def learn(self, state, action, reward, next_state, done):
            """Update Q-table using Q-learning update rule"""
            current_q = self.get_q_value(state, action)
            
            if done:
                max_next_q = 0
            else:
                max_next_q = np.max(self.q_table[next_state])
            
            # Q-learning update
            new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
            self.q_table[state][action] = new_q
            
            # Decay exploration rate
            if done:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        def train(self, episodes=1000):
            """Train the agent"""
            for episode in range(episodes):
                state = self.discretize_state(self.env.reset())
                total_reward = 0
                done = False
                
                while not done:
                    action = self.choose_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = self.discretize_state(next_state)
                    
                    self.learn(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                
                self.rewards.append(total_reward)
                
                if (episode + 1) % 100 == 0:
                    avg_reward = np.mean(self.rewards[-100:])
                    print(f"Episode {{episode+1}}, Avg Reward (last 100): {{avg_reward:.2f}}, Epsilon: {{self.epsilon:.3f}}")
            
            return self.rewards
        
        def plot_rewards(self):
            """Plot training rewards"""
            plt.figure(figsize=(10, 6))
            plt.plot(self.rewards, label='Episode Reward')
            
            # Plot moving average
            window_size = 100
            moving_avg = [np.mean(self.rewards[max(0, i-window_size):i+1]) 
                         for i in range(len(self.rewards))]
            plt.plot(moving_avg, label=f'Moving Avg ({{window_size}} episodes)', color='red')
            
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Q-Learning Training Progress')
            plt.legend()
            plt.grid(True)
            plt.savefig('qlearning_rewards.png')
            plt.close()
            print("Saved rewards plot to qlearning_rewards.png")
    
    class DQNAgent:
        def __init__(self, env, gamma=0.95, epsilon=1.0, epsilon_min=0.01, 
                     epsilon_decay=0.995, learning_rate=0.001, memory_size=2000):
            self.env = env
            self.state_size = env.observation_space.shape[0]
            self.action_size = env.action_space.n
            self.memory = deque(maxlen=memory_size)
            self.gamma = gamma  # Discount rate
            self.epsilon = epsilon  # Exploration rate
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
            self.learning_rate = learning_rate
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()
            self.rewards = []
        
        def _build_model(self):
            """Build neural network model"""
            model = Sequential()
            model.add(Dense(24, input_dim=self.state_size, activation='relu'))
            model.add(Dense(24, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
            return model
        
        def update_target_model(self):
            """Copy weights from main model to target model"""
            self.target_model.set_weights(self.model.get_weights())
        
        def remember(self, state, action, reward, next_state, done):
            """Store experience in memory"""
            self.memory.append((state, action, reward, next_state, done))
        
        def choose_action(self, state):
            """Epsilon-greedy action selection"""
            if np.random.rand() <= self.epsilon:
                return self.env.action_space.sample()
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])
        
        def replay(self, batch_size=32):
            """Train on batch from memory"""
            if len(self.memory) < batch_size:
                return
            
            minibatch = random.sample(self.memory, batch_size)
            states = np.array([x[0][0] for x in minibatch])
            actions = np.array([x[1] for x in minibatch])
            rewards = np.array([x[2] for x in minibatch])
            next_states = np.array([x[3][0] for x in minibatch])
            dones = np.array([x[4] for x in minibatch])
            
            # Predict Q-values for current and next states
            current_q = self.model.predict(states)
            next_q = self.target_model.predict(next_states)
            
            # Update Q-values using Bellman equation
            for i in range(batch_size):
                if dones[i]:
                    current_q[i][actions[i]] = rewards[i]
                else:
                    current_q[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_q[i])
            
            # Train model
            self.model.fit(states, current_q, epochs=1, verbose=0)
            
            # Decay exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        
        def train(self, episodes=500, batch_size=32, target_update_freq=10):
            """Train the agent"""
            for episode in range(episodes):
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])
                total_reward = 0
                done = False
                
                while not done:
                    action = self.choose_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    
                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                
                self.rewards.append(total_reward)
                self.replay(batch_size)
                
                # Periodically update target model
                if episode % target_update_freq == 0:
                    self.update_target_model()
                
                if (episode + 1) % 50 == 0:
                    avg_reward = np.mean(self.rewards[-50:])
                    print(f"Episode {{episode+1}}, Avg Reward (last 50): {{avg_reward:.2f}}, Epsilon: {{self.epsilon:.3f}}")
            
            return self.rewards
        
        def plot_rewards(self):
            """Plot training rewards"""
            plt.figure(figsize=(10, 6))
            plt.plot(self.rewards, label='Episode Reward')
            
            # Plot moving average
            window_size = 50
            moving_avg = [np.mean(self.rewards[max(0, i-window_size):i+1]) 
                         for i in range(len(self.rewards))]
            plt.plot(moving_avg, label=f'Moving Avg ({{window_size}} episodes)', color='red')
            
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('DQN Training Progress')
            plt.legend()
            plt.grid(True)
            plt.savefig('dqn_rewards.png')
            plt.close()
            print("Saved rewards plot to dqn_rewards.png")
        
        def test(self, episodes=10, render=True):
            """Test the trained agent"""
            test_rewards = []
            
            for episode in range(episodes):
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])
                total_reward = 0
                done = False
                
                while not done:
                    if render:
                        self.env.render()
                    action = np.argmax(self.model.predict(state)[0])
                    next_state, reward, done, _ = self.env.step(action)
                    state = np.reshape(next_state, [1, self.state_size])
                    total_reward += reward
                
                test_rewards.append(total_reward)
                print(f"Test Episode {{episode+1}}, Reward: {{total_reward:.1f}}")
            
            self.env.close()
            print(f"Average Test Reward: {{np.mean(test_rewards):.2f}}")
    
    def compare_rl_methods():
        """Compare Q-learning and DQN on CartPole"""
        env = gym.make('CartPole-v1')
        
        print("\\n=== Q-Learning ===")
        ql_agent = QLearningAgent(env)
        ql_rewards = ql_agent.train(episodes=1000)
        ql_agent.plot_rewards()
        
        print("\\n=== Deep Q-Network ===")
        dqn_agent = DQNAgent(env)
        dqn_rewards = dqn_agent.train(episodes=200)
        dqn_agent.plot_rewards()
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        
        # Q-learning rewards (smoothed)
        ql_window = 50
        ql_smooth = [np.mean(ql_rewards[max(0, i-ql_window):i+1]) 
                     for i in range(len(ql_rewards))]
        plt.plot(ql_smooth, label='Q-Learning')
        
        # DQN rewards (smoothed)
        dqn_window = 10
        dqn_smooth = [np.mean(dqn_rewards[max(0, i-dqn_window):i+1]) 
                      for i in range(len(dqn_rewards))]
        plt.plot(dqn_smooth, label='DQN')
        
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Comparison of Q-Learning and DQN')
        plt.legend()
        plt.grid(True)
        plt.savefig('rl_comparison.png')
        plt.close()
        print("Saved comparison plot to rl_comparison.png")
        
        # Test DQN
        print("\\nTesting DQN agent...")
        dqn_agent.test()
        
        env.close()
    
    if __name__ == '__main__':
        compare_rl_methods()
    ''')

    with open(filename, 'w') as f:
        f.write(content)


def generate_program_15(dir_name):
    """Generate a web application with Flask and React"""
    filename = os.path.join(dir_name, "flask_react_app.py")
    content = dedent(f'''\
    # Advanced Flask + React Web Application
    # Generated on {datetime.now()}
    # This program creates a full-stack web app with Flask backend and React frontend
    
    import os
    from flask import Flask, jsonify, request, send_from_directory
    from flask_cors import CORS
    import uuid
    from datetime import datetime
    
    # Initialize Flask app
    app = Flask(__name__, static_folder='../frontend/build')
    CORS(app)
    
    # Sample data storage (in a real app, use a database)
    posts = [
        {{'id': str(uuid.uuid4()), 'title': 'First Post', 'content': 'This is my first post!', 'date': '2023-01-01'}},
        {{'id': str(uuid.uuid4()), 'title': 'Second Post', 'content': 'Another interesting post.', 'date': '2023-01-02'}}
    ]
    
    users = [
        {{'id': '1', 'username': 'admin', 'password': 'admin'}}
    ]
    
    # Helper functions
    def find_post(post_id):
        return next((post for post in posts if post['id'] == post_id), None)
    
    def authenticate(username, password):
        return next((user for user in users if user['username'] == username and user['password'] == password), None)
    
    # API Routes
    @app.route('/api/posts', methods=['GET'])
    def get_posts():
        return jsonify(posts)
    
    @app.route('/api/posts/<post_id>', methods=['GET'])
    def get_post(post_id):
        post = find_post(post_id)
        if post is None:
            return jsonify({{'error': 'Post not found'}}), 404
        return jsonify(post)
    
    @app.route('/api/posts', methods=['POST'])
    def create_post():
        if not request.is_json:
            return jsonify({{'error': 'Request must be JSON'}}), 400
        
        data = request.get_json()
        if not data.get('title') or not data.get('content'):
            return jsonify({{'error': 'Title and content are required'}}), 400
        
        new_post = {{
            'id': str(uuid.uuid4()),
            'title': data['title'],
            'content': data['content'],
            'date': datetime.now().strftime('%Y-%m-%d')
        }}
        posts.append(new_post)
        return jsonify(new_post), 201
    
    @app.route('/api/posts/<post_id>', methods=['PUT'])
    def update_post(post_id):
        post = find_post(post_id)
        if post is None:
            return jsonify({{'error': 'Post not found'}}), 404
        
        if not request.is_json:
            return jsonify({{'error': 'Request must be JSON'}}), 400
        
        data = request.get_json()
        if 'title' in data:
            post['title'] = data['title']
        if 'content' in data:
            post['content'] = data['content']
        
        return jsonify(post)
    
    @app.route('/api/posts/<post_id>', methods=['DELETE'])
    def delete_post(post_id):
        global posts
        post = find_post(post_id)
        if post is None:
            return jsonify({{'error': 'Post not found'}}), 404
        
        posts = [p for p in posts if p['id'] != post_id]
        return jsonify({{'message': 'Post deleted'}}), 200
    
    @app.route('/api/login', methods=['POST'])
    def login():
        if not request.is_json:
            return jsonify({{'error': 'Request must be JSON'}}), 400
        
        data = request.get_json()
        user = authenticate(data.get('username'), data.get('password'))
        if user is None:
            return jsonify({{'error': 'Invalid credentials'}}), 401
        
        # In a real app, return a JWT token
        return jsonify({{'message': 'Login successful', 'user': {{'id': user['id'], 'username': user['username']}}}})
    
    # Serve React App
    @app.route('/', defaults={{'path': ''}})
    @app.route('/<path:path>')
    def serve(path):
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, 'index.html')
    
    if __name__ == '__main__':
        # Create React build directory if it doesn't exist
        if not os.path.exists(app.static_folder):
            os.makedirs(app.static_folder)
            with open(os.path.join(app.static_folder, 'index.html'), 'w') as f:
                f.write('''
                     <!DOCTYPE html >
                     < html lang="en" >
                     < head >
                     < meta charset="UTF-8" >
                     < meta name="viewport" content="width=device-width, initial-scale=1.0" >
                     < title > Flask + React App < /title >
                     < / head >
                     < body >
                     < div id="root" > </div >
                     < script >
                     // Simple React-like frontend for demonstration
                     document.addEventListener('DOMContentLoaded', async ()= > {
                         const root = document.getElementById('root');

                         // Fetch posts from API
                         const response = await fetch('/api/posts');
                         const posts = await response.json();

                         // Render posts
                         const postList = document.createElement('div');
                         postList.innerHTML = '<h1>Posts</h1>';

                         posts.forEach(post= > {
                             const postDiv = document.createElement('div');
                             postDiv.style.border = '1px solid #ccc';
                             postDiv.style.padding = '10px';
                             postDiv.style.margin = '10px 0';
                             postDiv.innerHTML = `
                             < h2 >${{post.title}} < /h2 >
                             < p >${{post.content}} < /p >
                             < small >${{post.date}} < /small > `;
                             postList.appendChild(postDiv);
                         });

                         root.appendChild(postList);

                         // Add a simple form to create new posts
                         const form = document.createElement('form');
                         form.innerHTML = `
                         < h2 > Create New Post < /h2 >
                         < input type = "text" id = "title" placeholder = "Title" required > <br >
                         < textarea id = "content" placeholder = "Content" required > </textarea > <br >
                         < button type = "submit" > Submit < /button >
                         `;

                         form.onsubmit= async (e) = > {
                             e.preventDefault();
                             const title = document.getElementById('title').value;
                             const content = document.getElementById('content').value;

                             await fetch('/api/posts', {
                                 method: 'POST',
                                 headers: {'Content-Type': 'application/json'},
                                 body: JSON.stringify({{title, content}})
                             });

                             alert('Post created! Refresh to see it.');
                         };

                         root.appendChild(form);
                     });
                     < /script >
                     < / body >
                     < / html >
                     ''')
        
        app.run(debug=True)
    ''')

    with open(filename, 'w') as f:
        f.write(content)


def generate_program_16(dir_name):
    """Generate a data pipeline with Airflow"""
    filename = os.path.join(dir_name, "data_pipeline.py")
    content = dedent(f'''\
    # Advanced Data Pipeline with Apache Airflow
    # Generated on {datetime.now()}
    # This program defines a complete ETL pipeline using Airflow
    
    from datetime import datetime, timedelta
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.email import EmailOperator
    from airflow.providers.postgres.operators.postgres import PostgresOperator
    from airflow.providers.amazon.aws.hooks.s3 import S3Hook
    from airflow.providers.postgres.hooks.postgres import PostgresHook
    import pandas as pd
    import logging
    
    default_args = {
        'owner': 'data_team',
        'depends_on_past': False,
        'start_date': datetime(2023, 1, 1),
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5)
    }
    
    dag = DAG(
        'advanced_data_pipeline',
        default_args=default_args,
        description='A complete ETL pipeline with data validation',
        schedule_interval='@daily',
        catchup=False,
        tags=['etl', 'data_processing']
    )
    
    def extract_data(**kwargs):
        """Extract data from source systems"""
        ti = kwargs['ti']
        execution_date = kwargs['execution_date']
        
        # Simulate extracting data from different sources
        logging.info(f"Extracting data for {{execution_date}}")
        
        # Source 1: API data
        api_data = [
            {'date': execution_date.strftime('%Y-%m-%d'), 'product': 'A', 'sales': 150},
            {'date': execution_date.strftime('%Y-%m-%d'), 'product': 'B', 'sales': 230},
            {'date': execution_date.strftime('%Y-%m-%d'), 'product': 'C', 'sales': 95}
        ]
        
        # Source 2: Database data
        db_data = [
            {'date': execution_date.strftime('%Y-%m-%d'), 'region': 'North', 'revenue': 1200},
            {'date': execution_date.strftime('%Y-%m-%d'), 'region': 'South', 'revenue': 1800},
            {'date': execution_date.strftime('%Y-%m-%d'), 'region': 'East', 'revenue': 950},
            {'date': execution_date.strftime('%Y-%m-%d'), 'region': 'West', 'revenue': 2100}
        ]
        
        # Push data to XCom for downstream tasks
        ti.xcom_push(key='api_data', value=api_data)
        ti.xcom_push(key='db_data', value=db_data)
        
        # Also save raw data to S3 for backup
        s3_hook = S3Hook(aws_conn_id='aws_default')
        
        api_df = pd.DataFrame(api_data)
        db_df = pd.DataFrame(db_data)
        
        api_csv = api_df.to_csv(index=False)
        db_csv = db_df.to_csv(index=False)
        
        s3_hook.load_string(
            string_data=api_csv,
            key=f'raw_data/api/{{execution_date}}.csv',
            bucket_name='data-pipeline-bucket',
            replace=True
        )
        
        s3_hook.load_string(
            string_data=db_csv,
            key=f'raw_data/db/{{execution_date}}.csv',
            bucket_name='data-pipeline-bucket',
            replace=True
        )
        
        logging.info("Data extraction and backup complete")
    
    def transform_data(**kwargs):
        """Transform and join data from different sources"""
        ti = kwargs['ti']
        execution_date = kwargs['execution_date']
        
        # Pull data from XCom
        api_data = ti.xcom_pull(task_ids='extract_data', key='api_data')
        db_data = ti.xcom_pull(task_ids='extract_data', key='db_data')
        
        # Convert to DataFrames
        sales_df = pd.DataFrame(api_data)
        revenue_df = pd.DataFrame(db_data)
        
        # Data cleaning and transformation
        sales_df['date'] = pd.to_datetime(sales_df['date'])
        revenue_df['date'] = pd.to_datetime(revenue_df['date'])
        
        # Calculate total sales by region (mapping product to region)
        product_region_map = {
            'A': 'North',
            'B': 'South',
            'C': 'East'
        }
        
        sales_df['region'] = sales_df['product'].map(product_region_map)
        sales_df.fillna('West', inplace=True)  # Default to West if no mapping
        
        # Aggregate data
        daily_sales = sales_df.groupby(['date', 'region'])['sales'].sum().reset_index()
        daily_revenue = revenue_df.groupby(['date', 'region'])['revenue'].sum().reset_index()
        
        # Join data
        final_df = pd.merge(
            daily_sales,
            daily_revenue,
            on=['date', 'region'],
            how='outer'
        )
        
        # Calculate additional metrics
        final_df['revenue_per_sale'] = final_df['revenue'] / final_df['sales']
        final_df.fillna(0, inplace=True)
        
        # Push transformed data to XCom
        ti.xcom_push(key='transformed_data', value=final_df.to_dict('records'))
        
        # Save transformed data to S3
        s3_hook = S3Hook(aws_conn_id='aws_default')
        transformed_csv = final_df.to_csv(index=False)
        
        s3_hook.load_string(
            string_data=transformed_csv,
            key=f'transformed_data/{{execution_date}}.csv',
            bucket_name='data-pipeline-bucket',
            replace=True
        )
        
        logging.info("Data transformation complete")
    
    def validate_data(**kwargs):
        """Validate data quality before loading"""
        ti = kwargs['ti']
        
        # Pull transformed data
        data = ti.xcom_pull(task_ids='transform_data', key='transformed_data')
        df = pd.DataFrame(data)
        
        # Data quality checks
        errors = []
        
        # Check 1: Negative values
        if (df['sales'] < 0).any():
            errors.append("Negative values found in sales")
        
        if (df['revenue'] < 0).any():
            errors.append("Negative values found in revenue")
        
        # Check 2: Missing values
        if df.isnull().values.any():
            errors.append("Missing values found in data")
        
        # Check 3: Revenue per sale outliers
        upper_limit = df['revenue_per_sale'].mean() + 3 * df['revenue_per_sale'].std()
        lower_limit = df['revenue_per_sale'].mean() - 3 * df['revenue_per_sale'].std()
        
        if ((df['revenue_per_sale'] > upper_limit) | (df['revenue_per_sale'] < lower_limit)).any():
            errors.append("Outliers detected in revenue per sale")
        
        # Push validation results
        if errors:
            ti.xcom_push(key='validation_errors', value=errors)
            raise ValueError("Data validation failed: " + ", ".join(errors))
        else:
            ti.xcom_push(key='validation_passed', value=True)
            logging.info("Data validation passed")
    
    def load_data(**kwargs):
        """Load validated data into data warehouse"""
        ti = kwargs['ti']
        
        # Check if validation passed
        validation_passed = ti.xcom_pull(task_ids='validate_data', key='validation_passed')
        if not validation_passed:
            raise ValueError("Cannot load data - validation failed")
        
        # Get transformed data
        data = ti.xcom_pull(task_ids='transform_data', key='transformed_data')
        df = pd.DataFrame(data)
        
        # Load to PostgreSQL
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
        conn = postgres_hook.get_conn()
        cursor = conn.cursor()
        
        # Create table if not exists
        create_table_sql = '''
                     CREATE TABLE IF NOT EXISTS daily_sales_metrics(
                         date DATE,
                         region VARCHAR(50),
                         sales NUMERIC,
                         revenue NUMERIC,
                         revenue_per_sale NUMERIC,
                         load_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                         PRIMARY KEY(date, region)
                     )
                     '''
        cursor.execute(create_table_sql)
        
        # Insert data
        for _, row in df.iterrows():
            insert_sql = f'''
                     INSERT INTO daily_sales_metrics(date, region, sales, revenue, revenue_per_sale)
                     VALUES('{{row['date']}}', '{{row['region']}}', {{row['sales']}}, {{row['revenue']}}, {{row['revenue_per_sale']}})
                     ON CONFLICT(date, region) DO UPDATE SET
                     sales=EXCLUDED.sales,
                     revenue=EXCLUDED.revenue,
                     revenue_per_sale=EXCLUDED.revenue_per_sale,
                     load_timestamp=CURRENT_TIMESTAMP
                     '''
            cursor.execute(insert_sql)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logging.info("Data loading complete")
    
    def generate_report(**kwargs):
        """Generate a data quality report"""
        ti = kwargs['ti']
        execution_date = kwargs['execution_date']
        
        # Check if validation had errors
        errors = ti.xcom_pull(task_ids='validate_data', key='validation_errors', default=None)
        
        report_content = f"Data Quality Report for {{execution_date.strftime('%Y-%m-%d')}}\\n\\n"
        
        if errors:
            report_content += "Status: FAILED\\n"
            report_content += "Errors found:\\n"
            for error in errors:
                report_content += f"- {{error}}\\n"
        else:
            report_content += "Status: PASSED\\n"
            report_content += "All data quality checks passed successfully.\\n"
        
        # Push report content to XCom for email task
        ti.xcom_push(key='report_content', value=report_content)
        
        # Also save report to S3
        s3_hook = S3Hook(aws_conn_id='aws_default')
        s3_hook.load_string(
            string_data=report_content,
            key=f'reports/{{execution_date}}.txt',
            bucket_name='data-pipeline-bucket',
            replace=True
        )
        
        logging.info("Report generation complete")
    
    # Define tasks
    extract_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
        provide_context=True,
        dag=dag
    )
    
    transform_task = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data,
        provide_context=True,
        dag=dag
    )
    
    validate_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        provide_context=True,
        dag=dag
    )
    
    load_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
        provide_context=True,
        dag=dag
    )
    
    report_task = PythonOperator(
        task_id='generate_report',
        python_callable=generate_report,
        provide_context=True,
        dag=dag
    )
    
    email_task = EmailOperator(
        task_id='send_email',
        to='data_team@example.com',
        subject='Data Pipeline Report - {{ ds }}',
        html_content='{{ ti.xcom_pull(task_ids="generate_report", key="report_content") }}',
        dag=dag
    )
    
    # Set up task dependencies
    extract_task >> transform_task >> validate_task >> load_task
    validate_task >> report_task >> email_task
    
    # Additional task to analyze data (runs in parallel with loading)
    analyze_task = PostgresOperator(
        task_id='analyze_data',
        postgres_conn_id='postgres_default',
        sql='''
                     ANALYZE daily_sales_metrics;

                     -- Create aggregate table for reporting
                     CREATE TABLE IF NOT EXISTS sales_trends AS
                     SELECT
                     date,
                     SUM(sales) as total_sales,
                     SUM(revenue) as total_revenue
                     FROM daily_sales_metrics
                     GROUP BY date
                     ORDER BY date;

                     -- Refresh materialized view
                     REFRESH MATERIALIZED VIEW CONCURRENTLY sales_region_trends;
                     ''',
        dag=dag
    )
    
    validate_task >> analyze_task
    ''')

    with open(filename, 'w') as f:
        f.write(content)


def generate_program_17(dir_name):
    """Generate a microservices architecture with FastAPI"""
    filename = os.path.join(dir_name, "microservices.py")
    content = dedent(f'''\
    # Advanced Microservices Architecture with FastAPI
    # Generated on {datetime.now()}
    # This program demonstrates a microservices system with 3 services and API gateway
    
    from fastapi import FastAPI, HTTPException, Depends
    from fastapi.security import OAuth2PasswordBearer
    from pydantic import BaseModel
    from typing import List, Optional
    import uuid
    import datetime
    import uvicorn
    from fastapi.middleware.cors import CORSMiddleware
    
    ###########################
    # User Service
    ###########################
    
    class User(BaseModel):
        id: str
        username: str
        email: str
        full_name: Optional[str] = None
        disabled: bool = False
    
    class UserInDB(User):
        hashed_password: str
    
    fake_users_db = {
        "johndoe": {
            "id": str(uuid.uuid4()),
            "username": "johndoe",
            "email": "johndoe@example.com",
            "full_name": "John Doe",
            "disabled": False,
            "hashed_password": "fakehashedsecret"
        },
        "alice": {
            "id": str(uuid.uuid4()),
            "username": "alice",
            "email": "alice@example.com",
            "full_name": "Alice Wonderson",
            "disabled": False,
            "hashed_password": "fakehashedsecret2"
        }
    }
    
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
    
    user_service = FastAPI()
    
    # Add CORS middleware
    user_service.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @user_service.post("/users/")
    async def create_user(username: str, email: str, password: str, full_name: Optional[str] = None):
        if username in fake_users_db:
            raise HTTPException(status_code=400, detail="Username already registered")
        
        user_dict = {
            "id": str(uuid.uuid4()),
            "username": username,
            "email": email,
            "full_name": full_name,
            "disabled": False,
            "hashed_password": f"fakehashed{{password}}"
        }
        fake_users_db[username] = user_dict
        return {"message": "User created successfully"}
    
    @user_service.get("/users/me")
    async def read_users_me(token: str = Depends(oauth2_scheme)):
        # In a real app, you would decode and verify the token
        # For simplicity, we'll just return the first user
        user = list(fake_users_db.values())[0]
        return User(**user)
    
    @user_service.get("/users/{username}")
    async def read_user(username: str):
        if username not in fake_users_db:
            raise HTTPException(status_code=404, detail="User not found")
        return User(**fake_users_db[username])
    
    ###########################
    # Product Service
    ###########################
    
    class Product(BaseModel):
        id: str
        name: str
        description: Optional[str] = None
        price: float
        category: str
    
    fake_products_db = [
        Product(
            id=str(uuid.uuid4()),
            name="Smartphone",
            description="Latest model smartphone",
            price=699.99,
            category="electronics"
        ),
        Product(
            id=str(uuid.uuid4()),
            name="Laptop",
            description="High performance laptop",
            price=1299.99,
            category="electronics"
        ),
        Product(
            id=str(uuid.uuid4()),
            name="Coffee Maker",
            description="Automatic coffee machine",
            price=89.99,
            category="home"
        )
    ]
    
    product_service = FastAPI()
    
    # Add CORS middleware
    product_service.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @product_service.get("/products/", response_model=List[Product])
    async def read_products(skip: int = 0, limit: int = 10):
        return fake_products_db[skip : skip + limit]
    
    @product_service.get("/products/{product_id}", response_model=Product)
    async def read_product(product_id: str):
        for product in fake_products_db:
            if product.id == product_id:
                return product
        raise HTTPException(status_code=404, detail="Product not found")
    
    @product_service.post("/products/", response_model=Product)
    async def create_product(product: Product):
        fake_products_db.append(product)
        return product
    
    @product_service.get("/products/category/{category}", response_model=List[Product])
    async def read_products_by_category(category: str):
        return [p for p in fake_products_db if p.category.lower() == category.lower()]
    
    ###########################
    # Order Service
    ###########################
    
    class OrderItem(BaseModel):
        product_id: str
        quantity: int
    
    class Order(BaseModel):
        id: str
        user_id: str
        items: List[OrderItem]
        order_date: str
        status: str  # "created", "processing", "shipped", "delivered", "cancelled"
    
    fake_orders_db = []
    
    order_service = FastAPI()
    
    # Add CORS middleware
    order_service.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @order_service.post("/orders/", response_model=Order)
    async def create_order(user_id: str, items: List[OrderItem]):
        order = Order(
            id=str(uuid.uuid4()),
            user_id=user_id,
            items=items,
            order_date=datetime.datetime.now().isoformat(),
            status="created"
        )
        fake_orders_db.append(order)
        return order
    
    @order_service.get("/orders/{order_id}", response_model=Order)
    async def read_order(order_id: str):
        for order in fake_orders_db:
            if order.id == order_id:
                return order
        raise HTTPException(status_code=404, detail="Order not found")
    
    @order_service.get("/orders/user/{user_id}", response_model=List[Order])
    async def read_user_orders(user_id: str):
        return [order for order in fake_orders_db if order.user_id == user_id]
    
    @order_service.put("/orders/{order_id}/status")
    async def update_order_status(order_id: str, status: str):
        for order in fake_orders_db:
            if order.id == order_id:
                order.status = status
                return {"message": "Order status updated"}
        raise HTTPException(status_code=404, detail="Order not found")
    
    ###########################
    # API Gateway
    ###########################
    
    from fastapi import FastAPI, Request
    import httpx
    
    api_gateway = FastAPI()
    
    # Service URLs (in a real app, these would be service discovery URLs)
    SERVICES = {
        "user": "http://localhost:8001",
        "product": "http://localhost:8002",
        "order": "http://localhost:8003"
    }
    
    @api_gateway.get("/{service}/{path:path}")
    async def proxy_get(service: str, path: str, request: Request):
        if service not in SERVICES:
            raise HTTPException(status_code=404, detail="Service not found")
        
        async with httpx.AsyncClient() as client:
            url = f"{{SERVICES[service]}}/{{path}}"
            response = await client.get(url, params=request.query_params)
            return response.json()
    
    @api_gateway.post("/{service}/{path:path}")
    async def proxy_post(service: str, path: str, request: Request):
        if service not in SERVICES:
            raise HTTPException(status_code=404, detail="Service not found")
        
        body = await request.json()
        async with httpx.AsyncClient() as client:
            url = f"{{SERVICES[service]}}/{{path}}"
            response = await client.post(url, json=body)
            return response.json()
    
    @api_gateway.put("/{service}/{path:path}")
    async def proxy_put(service: str, path: str, request: Request):
        if service not in SERVICES:
            raise HTTPException(status_code=404, detail="Service not found")
        
        body = await request.json()
        async with httpx.AsyncClient() as client:
            url = f"{{SERVICES[service]}}/{{path}}"
            response = await client.put(url, json=body)
            return response.json()
    
    ###########################
    # Main Application
    ###########################
    
    if __name__ == "__main__":
        import threading
        
        def run_service(app, port):
            uvicorn.run(app, host="0.0.0.0", port=port)
        
        # Start all services in separate threads
        services = [
            (user_service, 8001),
            (product_service, 8002),
            (order_service, 8003),
            (api_gateway, 8000)
        ]
        
        threads = []
        for app, port in services:
            thread = threading.Thread(target=run_service, args=(app, port))
            thread.start()
            threads.append(thread)
        
        print("Microservices running:")
        print("- User Service: http://localhost:8001")
        print("- Product Service: http://localhost:8002")
        print("- Order Service: http://localhost:8003")
        print("- API Gateway: http://localhost:8000")
        
        # Keep main thread alive
        for thread in threads:
            thread.join()
    ''')

    with open(filename, 'w') as f:
        f.write(content)


def generate_program_18(dir_name):
    """Generate a big data processing program with PySpark"""
    filename = os.path.join(dir_name, "big_data_processing.py")
    content = dedent(f'''\
    # Advanced Big Data Processing with PySpark
    # Generated on {datetime.now()}
    # This program demonstrates large-scale data processing with Spark
    
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import avg, dayofweek, month, year, sum, count
    from pyspark.sql.types import StructType, StructField, IntegerType, DateType, FloatType, StringType
    from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    import time
    
    class BigDataProcessor:
        def __init__(self):
            self.spark = SparkSession.builder \
                .appName("AdvancedDataProcessing") \
                .config("spark.executor.memory", "4g") \
                .config("spark.driver.memory", "2g") \
                .getOrCreate()\
                .config("spark.executor.memory", "4g") \
                .config("spark.driver.memory", "2g") \
                .getOrCreate()
            
            # Set log level to WARN to reduce verbose output
            self.spark.sparkContext.setLogLevel("WARN")
        
        def load_data(self, file_path):
            """Load data from CSV file"""
            print(f"Loading data from {{file_path}}...")
            start_time = time.time()
            
            # Define schema for better performance
            schema = StructType([
                StructField("id", IntegerType(), True),
                StructField("date", DateType(), True),
                StructField("store", IntegerType(), True),
                StructField("product", IntegerType(), True),
                StructField("sales", IntegerType(), True),
                StructField("revenue", FloatType(), True),
                StructField("temperature", FloatType(), True),
                StructField("holiday", StringType(), True),
                StructField("promotion", StringType(), True)
            ])
            
            df = self.spark.read.csv(
                file_path,
                header=True,
                schema=schema,
                dateFormat="yyyy-MM-dd"
            )
            
            print(f"Data loaded in {{time.time() - start_time:.2f}} seconds")
            print(f"Total records: {{df.count():,}}")
            
            return df
        
        def clean_data(self, df):
            """Clean and preprocess data"""
            print("\\nCleaning data...")
            start_time = time.time()
            
            # Handle missing values
            df = df.fillna({
                'temperature': df.select(avg('temperature')).first()[0],
                'holiday': 'None',
                'promotion': 'None'
            })
            
            # Add derived columns
            df = df.withColumn("day_of_week", dayofweek("date"))
            df = df.withColumn("month", month("date"))
            df = df.withColumn("year", year("date"))
            
            print(f"Data cleaned in {{time.time() - start_time:.2f}} seconds")
            return df
        
        def analyze_data(self, df):
            """Perform data analysis"""
            print("\\nAnalyzing data...")
            
            # Basic statistics
            print("\\nBasic Statistics:")
            df.select("sales", "revenue", "temperature").summary().show()
            
            # Sales by store
            print("\\nSales by Store:")
            df.groupBy("store").agg(
                sum("sales").alias("total_sales"),
                avg("sales").alias("avg_sales"),
                sum("revenue").alias("total_revenue")
            ).orderBy("total_sales", ascending=False).show()
            
            # Sales by product
            print("\\nSales by Product:")
            df.groupBy("product").agg(
                sum("sales").alias("total_sales"),
                avg("sales").alias("avg_sales")
            ).orderBy("total_sales", ascending=False).show(10)
            
            # Sales trend by month
            print("\\nMonthly Sales Trend:")
            df.groupBy("year", "month").agg(
                sum("sales").alias("total_sales")
            ).orderBy("year", "month").show(24)
        
        def prepare_features(self, df):
            """Prepare features for machine learning"""
            print("\\nPreparing features for ML...")
            start_time = time.time()
            
            # Convert categorical features
            holiday_indexer = StringIndexer(inputCol="holiday", outputCol="holidayIndex")
            promotion_indexer = StringIndexer(inputCol="promotion", outputCol="promotionIndex")
            
            # One-hot encode categorical features
            holiday_encoder = OneHotEncoder(inputCol="holidayIndex", outputCol="holidayVec")
            promotion_encoder = OneHotEncoder(inputCol="promotionIndex", outputCol="promotionVec")
            
            # Assemble all features
            assembler = VectorAssembler(
                inputCols=["store", "product", "temperature", "day_of_week", 
                          "month", "holidayVec", "promotionVec"],
                outputCol="features"
            )
            
            # Create pipeline
            pipeline = Pipeline(stages=[
                holiday_indexer,
                promotion_indexer,
                holiday_encoder,
                promotion_encoder,
                assembler
            ])
            
            # Fit and transform
            model = pipeline.fit(df)
            df_transformed = model.transform(df)
            
            print(f"Features prepared in {{time.time() - start_time:.2f}} seconds")
            return df_transformed
        
        def train_model(self, df):
            """Train a machine learning model"""
            print("\\nTraining ML model...")
            start_time = time.time()
            
            # Split data
            train, test = df.randomSplit([0.8, 0.2], seed=42)
            
            # Train Random Forest model
            rf = RandomForestClassifier(
                labelCol="sales",
                featuresCol="features",
                numTrees=50,
                maxDepth=10,
                seed=42
            )
            
            model = rf.fit(train)
            
            # Make predictions
            predictions = model.transform(test)
            
            # Evaluate model
            evaluator = MulticlassClassificationEvaluator(
                labelCol="sales",
                predictionCol="prediction",
                metricName="accuracy"
            )
            
            accuracy = evaluator.evaluate(predictions)
            print(f"Model accuracy: {{accuracy:.4f}}")
            print(f"Model trained in {{time.time() - start_time:.2f}} seconds")
            
            return model
        
        def process_large_dataset(self, input_path, output_path):
            """Process large dataset with optimizations"""
            print("\\nProcessing large dataset with optimizations...")
            start_time = time.time()
            
            # Read with partitioning
            df = self.spark.read.csv(
                input_path,
                header=True,
                inferSchema=True  # For large files, better to define schema explicitly
            )
            
            # Repartition for better parallelism
            df = df.repartition(100)
            
            # Cache frequently used DataFrame
            df.cache()
            
            # Perform transformations
            result = df.groupBy("store", "product").agg(
                sum("sales").alias("total_sales"),
                avg("sales").alias("avg_sales"),
                sum("revenue").alias("total_revenue"),
                count("*").alias("transaction_count")
            )
            
            # Write output in compressed format
            result.write.parquet(
                output_path,
                mode="overwrite",
                compression="snappy"
            )
            
            print(f"Processing completed in {{time.time() - start_time:.2f}} seconds")
            print(f"Results saved to {{output_path}}")
        
        def stop(self):
            """Stop Spark session"""
            self.spark.stop()
    
    def example_run():
        print("=== Big Data Processing Example ===")
        
        # Initialize processor
        processor = BigDataProcessor()
        
        try:
            # Load and analyze sample data
            df = processor.load_data("sales_data.csv")  # Replace with your data path
            df = processor.clean_data(df)
            processor.analyze_data(df)
            
            # Prepare features and train model
            df_features = processor.prepare_features(df)
            model = processor.train_model(df_features)
            
            # Process large dataset (commented out to prevent accidental runs)
            # processor.process_large_dataset("large_sales_data/*.csv", "output_results")
            
        finally:
            processor.stop()
    
    if __name__ == '__main__':
        example_run()
    ''')

    with open(filename, 'w') as f:
        f.write(content)


def generate_program_19(dir_name):
    """Generate a quantum computing simulation"""
    filename = os.path.join(dir_name, "quantum_computing.py")
    content = dedent(f'''\
    # Advanced Quantum Computing Simulation
    # Generated on {datetime.now()}
    # This program demonstrates quantum algorithms using Qiskit
    
    from qiskit import QuantumCircuit, Aer, execute
    from qiskit.visualization import plot_histogram, plot_bloch_multivector
    from qiskit.quantum_info import Statevector
    from qiskit.circuit.library import QFT
    import numpy as np
    import matplotlib.pyplot as plt
    
    class QuantumSimulator:
        def __init__(self):
            self.simulator = Aer.get_backend('qasm_simulator')
            self.statevector_simulator = Aer.get_backend('statevector_simulator')
            self.unitary_simulator = Aer.get_backend('unitary_simulator')
        
        def run_circuit(self, circuit, shots=1024):
            """Execute a quantum circuit"""
            result = execute(circuit, self.simulator, shots=shots).result()
            return result.get_counts(circuit)
        
        def get_statevector(self, circuit):
            """Get the statevector of a quantum circuit"""
            result = execute(circuit, self.statevector_simulator).result()
            return result.get_statevector()
        
        def get_unitary_matrix(self, circuit):
            """Get the unitary matrix of a quantum circuit"""
            result = execute(circuit, self.unitary_simulator).result()
            return result.get_unitary()
        
        def demo_superposition(self):
            """Demonstrate quantum superposition"""
            print("\\n=== Quantum Superposition ===")
            
            # Create a quantum circuit with 1 qubit
            qc = QuantumCircuit(1, 1)
            
            # Apply Hadamard gate to create superposition
            qc.h(0)
            
            # Measure the qubit
            qc.measure(0, 0)
            
            # Run the circuit
            counts = self.run_circuit(qc)
            print("Measurement results:", counts)
            
            # Visualize the statevector
            statevector = self.get_statevector(qc)
            from IPython.display import display
            display(Statevector(statevector).draw('latex'))
         
            # Plot Bloch sphere
            plot_bloch_multivector(statevector)
            plt.title("Qubit State on Bloch Sphere")
            plt.savefig('bloch_sphere.png')
            plt.close()
            print("Saved Bloch sphere visualization to bloch_sphere.png")
            
            return qc
        
        def demo_entanglement(self):
            """Demonstrate quantum entanglement (Bell state)"""
            print("\\n=== Quantum Entanglement ===")
            
            # Create a quantum circuit with 2 qubits
            qc = QuantumCircuit(2, 2)
            
            # Create Bell state
            qc.h(0)
            qc.cx(0, 1)
            
            # Measure both qubits
            qc.measure([0, 1], [0, 1])
            
            # Run the circuit
            counts = self.run_circuit(qc)
            print("Measurement results:", counts)
            
            # Visualize results
            plot_histogram(counts)
            plt.title("Bell State Measurement Results")
            plt.savefig('bell_state.png')
            plt.close()
            print("Saved Bell state results to bell_state.png")
            
            return qc
        
        def demo_quantum_teleportation(self):
            """Demonstrate quantum teleportation protocol"""
            print("\\n=== Quantum Teleportation ===")
            
            # Create a quantum circuit with 3 qubits
            qc = QuantumCircuit(3, 3)
            
            # Alice wants to teleport q0 to Bob
            
            # Step 1: Create entangled pair between Alice (q1) and Bob (q2)
            qc.h(1)
            qc.cx(1, 2)
            
            # Step 2: Prepare the state to be teleported (q0)
            qc.x(0)  # Put q0 in |1> state (can be any state)
            qc.barrier()
            
            # Step 3: Alice entangles q0 with her half of the Bell pair (q1)
            qc.cx(0, 1)
            qc.h(0)
            qc.barrier()
            
            # Step 4: Alice measures her qubits
            qc.measure([0, 1], [0, 1])
            qc.barrier()
            
            # Step 5: Bob applies gates based on measurement results
            qc.cx(1, 2)
            qc.cz(0, 2)
            
            # Measure Bob's qubit
            qc.measure(2, 2)
            
            # Run the circuit
            counts = self.run_circuit(qc)
            print("Measurement results:", counts)
            
            # Visualize results
            plot_histogram(counts)
            plt.title("Quantum Teleportation Results")
            plt.savefig('teleportation.png')
            plt.close()
            print("Saved teleportation results to teleportation.png")
            
            return qc
        
        def demo_quantum_fourier_transform(self, n_qubits=3):
            """Demonstrate Quantum Fourier Transform"""
            print(f"\\n=== Quantum Fourier Transform ({n_qubits} qubits) ===")
            
            # Create circuit
            qc = QuantumCircuit(n_qubits)
            
            # Prepare initial state (can be any state)
            for qubit in range(n_qubits):
                qc.h(qubit)
                qc.rx(np.pi/4, qubit)
            
            qc.barrier()
            
            # Apply QFT
            qft = QFT(n_qubits, do_swaps=True)
            qc.compose(qft, inplace=True)
            
            qc.barrier()
            
            # Measure all qubits
            qc.measure_all()
            
            # Run the circuit
            counts = self.run_circuit(qc)
            print("Measurement results:", counts)
            
            # Visualize results
            plot_histogram(counts)
            plt.title("QFT Measurement Results")
            plt.savefig('qft_results.png')
            plt.close()
            print("Saved QFT results to qft_results.png")
            
            # Get unitary matrix
            unitary = self.get_unitary_matrix(qc.decompose())
            print("\\nUnitary matrix shape:", unitary.shape)
            
            return qc
        
        def demo_grover_search(self, n_qubits=3, marked_item=5):
            """Demonstrate Grover's search algorithm"""
            print(f"\\n=== Grover's Search Algorithm ({n_qubits} qubits, marked item {marked_item}) ===")
            
            # Number of possible states
            N = 2**n_qubits
            
            # Optimal number of iterations
            iterations = int(np.round(np.pi/4 * np.sqrt(N)))
            
            # Create circuit
            qc = QuantumCircuit(n_qubits, n_qubits)
            
            # Initialize all qubits in superposition
            for qubit in range(n_qubits):
                qc.h(qubit)
            
            # Grover iterations
            for _ in range(iterations):
                # Oracle for the marked item
                # This marks the binary representation of marked_item
                binary_str = format(marked_item, f'0{n_qubits}b')
                
                # Apply X gates based on binary representation
                for qubit in range(n_qubits):
                    if binary_str[qubit] == '0':
                        qc.x(qubit)
                
                # Apply multi-controlled Z gate
                qc.h(n_qubits-1)
                qc.mct(list(range(n_qubits-1)), n_qubits-1)
                qc.h(n_qubits-1)
                
                # Undo X gates
                for qubit in range(n_qubits):
                    if binary_str[qubit] == '0':
                        qc.x(qubit)
                
                # Diffusion operator
                for qubit in range(n_qubits):
                    qc.h(qubit)
                    qc.x(qubit)
                
                qc.h(n_qubits-1)
                qc.mct(list(range(n_qubits-1)), n_qubits-1)
                qc.h(n_qubits-1)
                
                for qubit in range(n_qubits):
                    qc.x(qubit)
                    qc.h(qubit)
            
            # Measure all qubits
            qc.measure(range(n_qubits), range(n_qubits))
            
            # Run the circuit
            counts = self.run_circuit(qc, shots=2048)
            print("Measurement results:", counts)
            
            # Visualize results
            plot_histogram(counts)
            plt.title("Grover's Search Results")
            plt.savefig('grover_results.png')
            plt.close()
            print("Saved Grover's search results to grover_results.png")
            
            # Check success probability
            success_prob = counts.get(str(marked_item), 0) / 2048
            print(f"Success probability for item {marked_item}: {success_prob:.2%}")
            
            return qc
    
    if __name__ == '__main__':
        simulator = QuantumSimulator()
        
        # Run demonstrations
        simulator.demo_superposition()
        simulator.demo_entanglement()
        simulator.demo_quantum_teleportation()
        simulator.demo_quantum_fourier_transform()
        simulator.demo_grover_search()
    ''')

    with open(filename, 'w') as f:
        f.write(content)


def generate_program_20(dir_name):
    """Generate a cybersecurity program with encryption and hashing"""
    filename = os.path.join(dir_name, "cybersecurity.py")
    content = dedent(f'''\
    # Advanced Cybersecurity Program
    # Generated on {datetime.now()}
    # This program demonstrates various cybersecurity techniques
    
    import os
    import hashlib
    import hmac
    import secrets
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    import base64
    import getpass
    
    class CryptoUtils:
        @staticmethod
        def generate_salt(length=16):
            """Generate a random salt"""
            return secrets.token_bytes(length)
        
        @staticmethod
        def derive_key(password, salt, length=32, iterations=100000):
            """Derive a cryptographic key from a password using PBKDF2"""
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=length,
                salt=salt,
                iterations=iterations,
                backend=default_backend()
            )
            return kdf.derive(password.encode())
        
        @staticmethod
        def generate_aes_key(length=32):
            """Generate a random AES key"""
            return secrets.token_bytes(length)
        
        @staticmethod
        def aes_encrypt(data, key):
            """Encrypt data using AES-CBC"""
            # Generate a random IV
            iv = secrets.token_bytes(16)
            
            # Pad the data
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(data) + padder.finalize()
            
            # Create cipher and encrypt
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            
            return iv + ciphertext
        
        @staticmethod
        def aes_decrypt(encrypted_data, key):
            """Decrypt AES-CBC encrypted data"""
            # Extract IV and ciphertext
            iv = encrypted_data[:16]
            ciphertext = encrypted_data[16:]
            
            # Decrypt
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Unpad
            unpadder = padding.PKCS7(128).unpadder()
            data = unpadder.update(padded_data) + unpadder.finalize()
            
            return data
        
        @staticmethod
        def generate_rsa_key_pair(key_size=2048):
            """Generate RSA public/private key pair"""
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend()
            )
            public_key = private_key.public_key()
            return private_key, public_key
        
        @staticmethod
        def rsa_encrypt(data, public_key):
            """Encrypt data with RSA public key"""
            ciphertext = public_key.encrypt(
                data,
                asym_padding.OAEP(
                    mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return ciphertext
        
        @staticmethod
        def rsa_decrypt(ciphertext, private_key):
            """Decrypt data with RSA private key"""
            data = private_key.decrypt(
                ciphertext,
                asym_padding.OAEP(
                    mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return data
        
        @staticmethod
        def sign_data(data, private_key):
            """Sign data with private key"""
            signature = private_key.sign(
                data,
                asym_padding.PSS(
                    mgf=asym_padding.MGF1(hashes.SHA256()),
                    salt_length=asym_padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return signature
        
        @staticmethod
        def verify_signature(data, signature, public_key):
            """Verify signature with public key"""
            try:
                public_key.verify(
                    signature,
                    data,
                    asym_padding.PSS(
                        mgf=asym_padding.MGF1(hashes.SHA256()),
                        salt_length=asym_padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                return True
            except Exception:
                return False
        
        @staticmethod
        def save_private_key(private_key, filename, password=None):
            """Save private key to file with optional encryption"""
            encryption = (
                serialization.BestAvailableEncryption(password.encode()) 
                if password 
                else serialization.NoEncryption()
            )
            
            pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=encryption
            )
            
            with open(filename, 'wb') as f:
                f.write(pem)
        
        @staticmethod
        def save_public_key(public_key, filename):
            """Save public key to file"""
            pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            with open(filename, 'wb') as f:
                f.write(pem)
        
        @staticmethod
        def load_private_key(filename, password=None):
            """Load private key from file"""
            with open(filename, 'rb') as f:
                pem = f.read()
            
            return serialization.load_pem_private_key(
                pem,
                password=password.encode() if password else None,
                backend=default_backend()
            )
        
        @staticmethod
        def load_public_key(filename):
            """Load public key from file"""
            with open(filename, 'rb') as f:
                pem = f.read()
            
            return serialization.load_pem_public_key(
                pem,
                backend=default_backend()
            )
    
    class PasswordManager:
        def __init__(self):
            self.crypto = CryptoUtils()
            self.users = {}
        
        def create_user(self, username, password):
            """Create a new user with hashed password"""
            if username in self.users:
                raise ValueError("Username already exists")
            
            salt = self.crypto.generate_salt()
            hashed_password = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode(),
                salt,
                100000
            )
            
            self.users[username] = {
                'salt': salt,
                'hashed_password': hashed_password
            }
            print(f"User {username} created successfully")
        
        def authenticate(self, username, password):
            """Authenticate a user"""
            if username not in self.users:
                return False
            
            user_data = self.users[username]
            new_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode(),
                user_data['salt'],
                100000
            )
            
            return hmac.compare_digest(new_hash, user_data['hashed_password'])
    
    class SecureFileStorage:
        def __init__(self):
            self.crypto = CryptoUtils()
        
        def encrypt_file(self, input_file, output_file, password):
            """Encrypt a file with password-based encryption"""
            salt = self.crypto.generate_salt()
            key = self.crypto.derive_key(password, salt)
            
            with open(input_file, 'rb') as f:
                data = f.read()
            
            encrypted_data = self.crypto.aes_encrypt(data, key)
            
            with open(output_file, 'wb') as f:
                f.write(salt + encrypted_data)
            
            print(f"File encrypted and saved to {output_file}")
        
        def decrypt_file(self, input_file, output_file, password):
            """Decrypt a password-encrypted file"""
            with open(input_file, 'rb') as f:
                data = f.read()
            
            salt = data[:16]
            encrypted_data = data[16:]
            
            key = self.crypto.derive_key(password, salt)
            decrypted_data = self.crypto.aes_decrypt(encrypted_data, key)
            
            with open(output_file, 'wb') as f:
                f.write(decrypted_data)
            
            print(f"File decrypted and saved to {output_file}")
    
    def demonstrate_crypto_functions():
        print("=== Cryptographic Functions Demonstration ===")
        crypto = CryptoUtils()
        
        # 1. Symmetric Encryption (AES)
        print("\\n1. AES Encryption:")
        key = crypto.generate_aes_key()
        message = b"Secret message to encrypt with AES"
        encrypted = crypto.aes_encrypt(message, key)
        decrypted = crypto.aes_decrypt(encrypted, key)
        
        print(f"Original: {message}")
        print(f"Encrypted: {base64.b64encode(encrypted).decode()}")
        print(f"Decrypted: {decrypted.decode()}")
        
        # 2. Password-based key derivation
        print("\\n2. Password-based Key Derivation:")
        password = "my_secure_password"
        salt = crypto.generate_salt()
        derived_key = crypto.derive_key(password, salt)
        print(f"Derived key: {base64.b64encode(derived_key).decode()}")
        
        # 3. Asymmetric Encryption (RSA)
        print("\\n3. RSA Encryption:")
        private_key, public_key = crypto.generate_rsa_key_pair()
        message = b"Secret message for RSA"
        encrypted = crypto.rsa_encrypt(message, public_key)
        decrypted = crypto.rsa_decrypt(encrypted, private_key)
        
        print(f"Original: {message}")
        print(f"Encrypted: {base64.b64encode(encrypted).decode()}")
        print(f"Decrypted: {decrypted.decode()}")
        
        # 4. Digital Signatures
        print("\\n4. Digital Signatures:")
        document = b"Important document to sign"
        signature = crypto.sign_data(document, private_key)
        is_valid = crypto.verify_signature(document, signature, public_key)
        
        print(f"Document: {document}")
        print(f"Signature: {base64.b64encode(signature).decode()}")
        print(f"Signature valid: {is_valid}")
        
        # 5. Key Serialization
        print("\\n5. Key Serialization:")
        crypto.save_private_key(private_key, "private_key.pem", "key_password")
        crypto.save_public_key(public_key, "public_key.pem")
        print("Keys saved to private_key.pem and public_key.pem")
        
        crypto.load_private_key("private_key.pem", "key_password")
        crypto.load_public_key("public_key.pem")
        print("Keys loaded successfully")
 
    def demonstrate_password_manager():
        print("\\n=== Password Manager Demonstration ===")
        manager = PasswordManager()
        
        # Create users
        manager.create_user("alice", "AlicePassword123!")
        manager.create_user("bob", "BobSecurePassword456!")
        
        # Test authentication
        print("\\nAuthentication tests:")
        print("Alice with correct password:", manager.authenticate("alice", "AlicePassword123!"))
        print("Alice with wrong password:", manager.authenticate("alice", "wrong"))
        print("Bob with correct password:", manager.authenticate("bob", "BobSecurePassword456!"))
        print("Nonexistent user:", manager.authenticate("eve", "anything"))
    
    def demonstrate_secure_storage():
        print("\\n=== Secure File Storage Demonstration ===")
        storage = SecureFileStorage()
        
        # Create a test file
        with open("test_file.txt", "w") as f:
            f.write("This is a secret file that needs to be encrypted.")
        
        # Encrypt and decrypt
        storage.encrypt_file("test_file.txt", "encrypted_file.bin", "file_password")
        storage.decrypt_file("encrypted_file.bin", "decrypted_file.txt", "file_password")
        
        # Show results
        print("\\nFile contents:")
        with open("decrypted_file.txt", "r") as f:
            print(f.read())
    
    if __name__ == '__main__':
        demonstrate_crypto_functions()
        demonstrate_password_manager()
        demonstrate_secure_storage()


    with open(filename, 'w') as f:
        f.write(content)


def main():
    """Main function to generate all program files"""
    dir_name = create_directory()

    # List of generator functions
    generators = [
        generate_program_1,
        generate_program_2,
        generate_program_3,
        generate_program_4,
        generate_program_5,
        generate_program_6,
        generate_program_7,
        generate_program_8,
        generate_program_9,
        generate_program_10,
        generate_program_11,
        generate_program_12,
        generate_program_13,
        generate_program_14,
        generate_program_15,
        generate_program_16,
        generate_program_17,
        generate_program_18,
        generate_program_19,
        generate_program_20
    ]

    # Generate all programs
    for i, generator in enumerate(generators, 1):
        print(f"Generating program {i}...")
        generator(dir_name)

       print(f"\nSuccessfully generated 20 different Python programs in the '{dir_name}' directory.")

if __name__ == "__main__":
    main()
