# src/attribute_schema.py
"""
Schema ánh xạ 601 classes của Open Images V7 (oidv7-class-descriptions-boxable.csv)
sang các nhóm Attribute ổn định nhất cho bài toán High-Resolution / Multi-Image Small Object Visual Search.

Cơ chế class_overrides: mỗi group có thể định nghĩa `class_overrides` để ghi đè
allowed_values cho TỪNG class riêng, giải quyết vấn đề group trộn nhiều loại vật thể:
- Giá trị là list  -> thay thế vocab mặc định của group cho attribute đó.
- Giá trị là None  -> bỏ hẳn attribute đó khỏi class (attribute không áp dụng được).
- Attribute chưa có trong stable_attributes của group -> tự động thêm vào (additive).
"""

from typing import Dict, List, Optional, Any

# ==============================================================================
# 1. DANH SÁCH GIÁ TRỊ CẤM (STOP-VALUES) - Tuyệt đối không dùng để tạo câu hỏi
# ==============================================================================
STOP_ATTRIBUTE_VALUES = {
    "solid", "plain", "normal", "standard", "regular", "irregular",
    "unknown", "none", "null", "unclear", "other", "mixed", "various"
}

# ==============================================================================
# 2. TỪ ĐIỂN TỪ VỰNG CHUẨN CHO TỪNG ATTRIBUTE (GLOBAL VOCABULARY)
# ==============================================================================
GLOBAL_ATTRIBUTE_VOCAB = {
    "color": [
        "red", "blue", "green", "yellow", "orange", "purple", "pink",
        "brown", "black", "white", "grey", "silver", "gold", "beige", "teal"
    ],
    "pattern": [
        "striped", "checkered", "plaid", "polka dot", "floral",
        "spotted", "camouflage", "graphic"
    ],
    "shape": [
        "round", "square", "rectangular", "oval", "triangular",
        "octagonal", "diamond", "cylindrical", "heart-shaped"
    ],
    "material": [
        "wooden", "metal", "plastic", "glass", "fabric",
        "leather", "ceramic", "wicker", "stone", "paper/cardboard"
    ],
    "orientation": [
        "facing left", "facing right", "front view", "rear view"
    ]
    # Lưu ý: "pose" chỉ được nhóm animals_and_birds sử dụng và nhóm đó tự định nghĩa
    # vocab riêng (gộp "walking/running"), nên không đặt ở đây để tránh lệch giá trị.
}

# ==============================================================================
# 3. PHÂN NHÓM CÁC CLASS HỢP LỆ & ATTRIBUTE ỔN ĐỊNH TƯƠNG ỨNG
#    (Tên class khớp chính xác 100% với cột DisplayName trong CSV)
# ==============================================================================
OIDV7_GROUP_SCHEMA: Dict[str, Dict[str, Any]] = {

    # NHÓM 1: Phương tiện đường bộ -> CẤM hỏi shape/material (tránh lỗi "rectangular plastic car")
    "land_vehicles": {
        "classes": [
            "Ambulance", "Bicycle", "Bus", "Car", "Cart", "Golf cart",
            "Limousine", "Motorcycle", "Segway", "Snowmobile", "Snowplow",
            "Tank", "Taxi", "Train", "Truck", "Unicycle", "Van", "Wheelchair"
        ],
        "stable_attributes": ["color", "orientation", "placement"],
        "allowed_values": {
            "color": GLOBAL_ATTRIBUTE_VOCAB["color"],
            "orientation": GLOBAL_ATTRIBUTE_VOCAB["orientation"],
            "placement": ["parked", "on road", "indoors"]
        },
        # subtype chỉ áp dụng cho một số class -> định nghĩa riêng từng class
        "class_overrides": {
            "Car": {"subtype": ["sedan", "suv", "sports car", "convertible", "pickup truck", "minivan", "vintage car", "race car"]},
            "Taxi": {"subtype": ["sedan", "suv", "minivan"]},
            "Bus": {"subtype": ["city bus", "coach bus", "school bus", "minibus", "double-decker bus"]},
            "Truck": {"subtype": ["pickup truck", "delivery truck", "semi-truck", "dump truck", "fire truck", "garbage truck"]},
            "Van": {"subtype": ["minivan", "cargo van", "passenger van", "camper van"]},
            "Limousine": {"subtype": ["stretch limousine", "classic sedan"]},
            "Motorcycle": {"subtype": ["sport bike", "cruiser", "dirt bike", "scooter", "touring bike"]},
            "Bicycle": {"subtype": ["road bike", "mountain bike", "city bike", "BMX bike", "folding bike"]},
            "Train": {
                "subtype": ["passenger train", "freight train", "locomotive", "tram"],
                "placement": ["on tracks", "at station"]
            }
        }
    },

    # NHÓM 2: Tàu thuyền & Máy bay
    "water_and_air_vehicles": {
        "classes": [
            "Aircraft", "Barge", "Boat", "Canoe", "Fixed-wing aircraft",
            "Gondola", "Helicopter", "Jet ski", "Rocket", "Submarine", "Watercraft"
        ],
        "stable_attributes": ["color", "placement", "orientation"],
        "allowed_values": {
            "color": GLOBAL_ATTRIBUTE_VOCAB["color"],
            "placement": ["in water", "on land", "in sky", "on runway"],
            "orientation": GLOBAL_ATTRIBUTE_VOCAB["orientation"]
        },
        # Tách subtype thuyền / máy bay theo từng class (tránh tổ hợp "Canoe + jet")
        "class_overrides": {
            "Aircraft": {"subtype": ["airliner", "biplane", "jet", "propeller plane", "glider"]},
            "Fixed-wing aircraft": {"subtype": ["airliner", "biplane", "jet", "propeller plane", "glider"]},
            "Boat": {"subtype": ["sailboat", "motorboat", "fishing boat", "rowboat", "speedboat"]},
            "Watercraft": {"subtype": ["sailboat", "motorboat", "fishing boat", "rowboat", "speedboat", "yacht", "kayak"]}
        }
    },

    # NHÓM 3: Bàn các loại -> Hình dáng mặt bàn (round/square/rectangular) và chất liệu cực kỳ ổn định
    "tables_and_desks": {
        "classes": [
            "Billiard table", "Coffee table", "Countertop", "Desk",
            "Kitchen & dining room table", "Nightstand", "Table"
        ],
        "stable_attributes": ["color", "shape", "material"],
        "allowed_values": {
            "color": GLOBAL_ATTRIBUTE_VOCAB["color"],
            "shape": ["round", "square", "rectangular", "oval"],
            "material": ["wooden", "metal", "glass", "plastic", "stone"]
        }
    },

    # NHÓM 4: Ghế & Sofa -> Hỏi color, material, subtype, pattern rất tự nhiên
    "seating_furniture": {
        "classes": [
            "Bench", "Chair", "Couch", "Infant bed", "Loveseat",
            "Sofa bed", "Stool", "Studio couch", "Training bench"
        ],
        "stable_attributes": ["color", "material", "pattern"],
        "allowed_values": {
            "color": GLOBAL_ATTRIBUTE_VOCAB["color"],
            "material": ["wooden", "metal", "plastic", "fabric", "leather", "wicker"],
            "pattern": GLOBAL_ATTRIBUTE_VOCAB["pattern"]
        },
        # subtype chỉ hợp lệ với ghế/couch đúng nghĩa (Infant bed, Training bench... không có)
        "class_overrides": {
            "Chair": {"subtype": ["armchair", "office chair", "folding chair", "dining chair", "rocking chair", "recliner"]},
            "Bench": {"subtype": ["park bench", "garden bench"]},
            "Stool": {"subtype": ["bar stool", "step stool"]},
            "Couch": {"subtype": ["sectional sofa", "chesterfield", "futon", "sofa bed"]}
        }
    },

    # NHÓM 5: Tủ, Kệ, Giường
    "storage_and_beds": {
        "classes": [
            "Bathroom cabinet", "Bed", "Bookcase", "Cabinetry", "Chest of drawers",
            "Closet", "Cupboard", "Dog bed", "Drawer", "Filing cabinet",
            "Shelf", "Spice rack", "Wardrobe", "Wine rack"
        ],
        "stable_attributes": ["color", "material", "state"],
        "allowed_values": {
            "color": GLOBAL_ATTRIBUTE_VOCAB["color"],
            "material": ["wooden", "metal", "plastic", "glass"],
            "state": ["open", "closed", "full", "empty"]
        },
        # full/empty không áp dụng cho giường; giường có state riêng (made/unmade)
        "class_overrides": {
            "Bed": {"state": ["made", "unmade"]},
            "Dog bed": {"state": None}
        }
    },

    # NHÓM 6: Quần áo & Trang phục -> Rất mạnh về Color và Pattern (kẻ sọc, caro, chấm bi...)
    "clothing_apparel": {
        "classes": [
            "Belt", "Brassiere", "Coat", "Dress", "Glove", "Jacket",
            "Jeans", "Miniskirt", "Scarf", "Shirt", "Shorts", "Skirt",
            "Sock", "Sports uniform", "Suit", "Swimwear", "Tie", "Trousers"
        ],
        "stable_attributes": ["color", "pattern", "placement"],
        "allowed_values": {
            "color": GLOBAL_ATTRIBUTE_VOCAB["color"],
            "pattern": GLOBAL_ATTRIBUTE_VOCAB["pattern"],
            "placement": ["worn by person", "hanging", "folded", "on mannequin"]
        }
    },

    # NHÓM 7: Mũ, Nón, Kính -> Rất mạnh về Color, Subtype (baseball cap, sun hat...) và Placement
    "headwear_and_eyewear": {
        "classes": [
            "Bicycle helmet", "Cowboy hat", "Crown", "Fedora", "Football helmet",
            "Glasses", "Goggles", "Hat", "Helmet", "Sombrero",
            "Sun hat", "Sunglasses", "Swim cap", "Tiara"
        ],
        "stable_attributes": ["color", "pattern", "placement"],
        "allowed_values": {
            "color": GLOBAL_ATTRIBUTE_VOCAB["color"],
            "pattern": GLOBAL_ATTRIBUTE_VOCAB["pattern"],
            "placement": ["worn on head", "held in hand", "on table/surface", "hanging"]
        },
        # Tách subtype mũ / kính; kính đeo placement riêng ("worn on face")
        "class_overrides": {
            "Hat": {"subtype": ["baseball cap", "beanie", "beret", "sun hat", "bucket hat", "straw hat", "top hat", "fedora"]},
            "Helmet": {"subtype": ["bike helmet", "motorcycle helmet", "hard hat", "climbing helmet"]},
            "Glasses": {
                "subtype": ["eyeglasses", "sunglasses", "reading glasses", "safety glasses"],
                "placement": ["worn on face", "held in hand", "on table/surface"]
            },
            "Sunglasses": {
                "subtype": ["aviator sunglasses", "wayfarer sunglasses", "round sunglasses", "sport sunglasses"],
                "placement": ["worn on face", "held in hand", "on table/surface"]
            },
            "Goggles": {
                "subtype": ["swimming goggles", "ski goggles", "safety goggles"],
                "placement": ["worn on face", "held in hand", "on table/surface"]
            }
        }
    },

    # NHÓM 8: Túi xách, Balo, Vali, Giày dép
    "bags_and_footwear": {
        "classes": [
            "Backpack", "Boot", "Briefcase", "Handbag", "High heels",
            "Luggage and bags", "Pencil case", "Plastic bag",
            "Roller skates", "Sandal", "Suitcase"
        ],
        "stable_attributes": ["color", "pattern", "placement", "state"],
        "allowed_values": {
            "color": GLOBAL_ATTRIBUTE_VOCAB["color"],
            "pattern": GLOBAL_ATTRIBUTE_VOCAB["pattern"],
            "placement": ["worn/carried", "on ground/floor", "on chair/table"],
            "state": ["open", "closed"]
        },
        # Tách subtype túi / giày; giày dép không có state open/closed
        "class_overrides": {
            "Luggage and bags": {"subtype": ["backpack", "duffel bag", "rolling suitcase", "tote bag", "handbag"]},
            "Handbag": {"subtype": ["tote bag", "shoulder bag", "clutch", "crossbody bag"]},
            "Suitcase": {"subtype": ["rolling suitcase", "hard-shell suitcase", "carry-on suitcase"]},
            "Boot": {"subtype": ["ankle boots", "knee-high boots", "rain boots", "work boots", "cowboy boots"], "state": None},
            "High heels": {"subtype": ["pumps", "stilettos", "wedge heels", "block heels"], "state": None},
            "Sandal": {"subtype": ["flip-flops", "strappy sandals", "slides", "sport sandals"], "state": None},
            "Roller skates": {"subtype": ["inline skates", "quad skates"], "state": None}
        }
    },

    # NHÓM 9: Ô dù, Lều, Cờ, Gối, Rèm -> Rất mạnh về State (open/closed/folded), Pattern, Color
    "umbrellas_tents_flags_textiles": {
        "classes": [
            "Curtain", "Flag", "Parachute", "Pillow", "Tent",
            "Towel", "Umbrella", "Window blind"
        ],
        "stable_attributes": ["color", "pattern", "state", "shape", "placement"],
        "allowed_values": {
            "color": GLOBAL_ATTRIBUTE_VOCAB["color"],
            "pattern": GLOBAL_ATTRIBUTE_VOCAB["pattern"],
            "state": ["open", "closed", "folded"],
            "shape": ["round", "square", "rectangular", "triangular", "dome-shaped"],
            "placement": ["mounted on pole", "held by person", "on ground", "hanging on wall"]
        }
    },

    # NHÓM 10: Biển báo, Đồng hồ, Khung tranh, Gương, Cửa -> Rất mạnh về Shape hình học & State
    "signs_clocks_frames_doors": {
        "classes": [
            "Alarm clock", "Billboard", "Clock", "Digital clock", "Door",
            "Mirror", "Parking meter", "Picture frame", "Poster", "Scoreboard",
            "Stop sign", "Traffic light", "Traffic sign", "Vehicle registration plate",
            "Wall clock", "Watch", "Whiteboard", "Window"
        ],
        "stable_attributes": ["color", "shape", "placement"],
        "allowed_values": {
            "color": GLOBAL_ATTRIBUTE_VOCAB["color"],
            "shape": ["round", "square", "rectangular", "triangular", "octagonal", "diamond", "oval"],
            "placement": ["hanging on wall", "mounted on pole", "on table/desk", "on building facade"]
        },
        # state chỉ hợp lệ với một số class: cửa (open/closed), đèn tín hiệu (signal), biển điện tử (lit)
        "class_overrides": {
            "Door": {"state": ["open", "closed"]},
            "Window": {"state": ["open", "closed"]},
            "Traffic light": {"state": ["lit/on", "unlit/off", "red signal", "green signal", "yellow signal"]},
            "Billboard": {"state": ["lit/on", "unlit/off"]},
            "Scoreboard": {"state": ["lit/on", "unlit/off"]},
            "Digital clock": {"state": ["lit/on", "unlit/off"]}
        }
    },

    # NHÓM 11: Đồ chứa, Chai lọ, Ly cốc, Bát đĩa, Thùng
    "containers_and_tableware": {
        "classes": [
            "Barrel", "Beaker", "Beer", "Bottle", "Bowl", "Box", "Cake stand",
            "Cocktail shaker", "Coffee cup", "Container", "Drinking straw",
            "Flowerpot", "Frying pan", "Jug", "Kettle", "Measuring cup",
            "Mixing bowl", "Mug", "Picnic basket", "Pitcher (Container)",
            "Plate", "Platter", "Salt and pepper shakers", "Saucer",
            "Serving tray", "Slow cooker", "Teapot", "Tin can", "Vase",
            "Waste container", "Wine glass", "Wok"
        ],
        "stable_attributes": ["color", "shape", "pattern", "state"],
        "allowed_values": {
            "color": GLOBAL_ATTRIBUTE_VOCAB["color"],
            "shape": ["round", "square", "rectangular", "cylindrical", "oval"],
            "pattern": GLOBAL_ATTRIBUTE_VOCAB["pattern"],
            "state": ["full", "empty", "open", "closed"]
        },
        # subtype cũ trộn ~27 class khác nhau -> chỉ giữ cho class thực sự có phân loại rõ
        "class_overrides": {
            "Bottle": {"subtype": ["plastic bottle", "glass bottle", "water bottle", "wine bottle"]},
            "Box": {"subtype": ["cardboard box", "shoe box", "gift box", "storage box"]},
            "Container": {"subtype": ["box", "jar", "bucket", "basket", "bin", "can"]},
            "Waste container": {"subtype": ["trash bin", "dumpster", "recycling bin"]},
            "Drinking straw": {"state": None}
        }
    },

    # NHÓM 12: Đèn & Thiết bị Điện tử / Gia dụng -> Rất mạnh về State (lit/on vs unlit/off) & Color
    "lamps_and_electronics": {
        "classes": [
            "Blender", "Calculator", "Camera", "Candle", "Cassette deck", "Ceiling fan",
            "Coffeemaker", "Computer keyboard", "Computer monitor", "Computer mouse",
            "Corded phone", "Dishwasher", "Fax", "Flashlight", "Food processor",
            "Gas stove", "Hair dryer", "Hand dryer", "Headphones", "Heater",
            "Humidifier", "Ipod", "Lamp", "Lantern", "Laptop", "Light bulb",
            "Light switch", "Mechanical fan", "Microphone", "Microwave oven",
            "Mixer", "Mobile phone", "Oven", "Printer", "Refrigerator",
            "Remote control", "Scale", "Sewing machine", "Street light",
            "Tablet computer", "Telephone", "Television", "Toaster",
            "Torch", "Tripod", "Waffle iron", "Washing machine", "Wood-burning stove"
        ],
        "stable_attributes": ["color", "state", "placement"],
        "allowed_values": {
            "color": GLOBAL_ATTRIBUTE_VOCAB["color"],
            "state": ["lit/on", "unlit/off", "open", "closed"],
            "placement": ["on table/desk", "hanging from ceiling", "mounted on wall/pole", "held in hand", "on floor"]
        }
    },

    # NHÓM 13: Động vật & Chim chóc -> Rất mạnh về Pose (standing, sitting, flying...), Orientation & Color
    "animals_and_birds": {
        "classes": [
            "Alpaca", "Ant", "Antelope", "Armadillo", "Bat (Animal)", "Bear", "Bee",
            "Beetle", "Bird", "Blue jay", "Brown bear", "Bull", "Butterfly", "Camel",
            "Canary", "Cat", "Caterpillar", "Cattle", "Centipede", "Cheetah",
            "Chicken", "Crab", "Crocodile", "Deer", "Dinosaur", "Dog", "Dolphin",
            "Dragonfly", "Duck", "Eagle", "Elephant", "Falcon", "Fish", "Fox",
            "Frog", "Giraffe", "Goat", "Goldfish", "Goose", "Hamster", "Harbor seal",
            "Hedgehog", "Hippopotamus", "Horse", "Isopod", "Jaguar (Animal)",
            "Jellyfish", "Kangaroo", "Koala", "Ladybug", "Leopard", "Lion", "Lizard",
            "Lobster", "Lynx", "Magpie", "Monkey", "Mouse", "Mule", "Ostrich",
            "Otter", "Owl", "Oyster", "Panda", "Parrot", "Penguin", "Pig",
            "Polar bear", "Porcupine", "Rabbit", "Raccoon", "Raven", "Rays and skates",
            "Red panda", "Rhinoceros", "Scorpion", "Sea lion", "Sea turtle",
            "Seahorse", "Shark", "Sheep", "Shrimp", "Skunk", "Snail", "Snake",
            "Sparrow", "Spider", "Squid", "Squirrel", "Starfish", "Swan", "Tick",
            "Tiger", "Tortoise", "Turkey", "Turtle", "Whale", "Woodpecker", "Worm", "Zebra"
        ],
        "stable_attributes": ["color", "pose", "orientation", "pattern", "placement"],
        "allowed_values": {
            "color": GLOBAL_ATTRIBUTE_VOCAB["color"],
            "pose": ["standing", "sitting", "lying down", "walking/running", "flying", "swimming", "perched"],
            "orientation": GLOBAL_ATTRIBUTE_VOCAB["orientation"],
            "pattern": ["spotted", "striped"],
            "placement": ["on grass/ground", "in water", "on branch/tree", "in sky", "indoors"]
        }
    },

    # NHÓM 14: Bóng, Dụng cụ thể thao, Nhạc cụ & Đồ chơi
    "sports_music_and_toys": {
        "classes": [
            "Accordion", "Balance beam", "Ball (Object)", "Balloon", "Banjo",
            "Baseball bat", "Baseball glove", "Bow and arrow", "Bowling equipment",
            "Cello", "Chime", "Cricket ball", "Dice", "Doll", "Drum", "Dumbbell",
            "Flute", "Flying disc", "Football", "French horn", "Golf ball",
            "Guitar", "Harmonica", "Harp", "Harpsichord", "Horizontal bar",
            "Indoor rower", "Kite", "Maraca", "Musical keyboard", "Oboe",
            "Organ (Musical Instrument)", "Paddle", "Personal flotation device",
            "Piano", "Punching bag", "Racket", "Rugby ball", "Saxophone",
            "Skateboard", "Ski", "Snowboard", "Snowman", "Stationary bicycle",
            "Surfboard", "Table tennis racket", "Teddy bear", "Tennis ball",
            "Tennis racket", "Treadmill", "Trombone", "Trumpet", "Violin", "Volleyball (Ball)"
        ],
        "stable_attributes": ["color", "pattern", "shape", "placement"],
        "allowed_values": {
            "color": GLOBAL_ATTRIBUTE_VOCAB["color"],
            "pattern": GLOBAL_ATTRIBUTE_VOCAB["pattern"],
            "shape": ["round", "oval", "star-shaped", "heart-shaped", "diamond"],
            "placement": ["held/played by person", "on ground/floor", "in air", "on stand/wall"]
        }
    },

    # NHÓM 15: Trái cây, Rau củ & Đồ ăn -> Mạnh về Color, State (whole vs sliced/cut), Placement
    "fruits_vegetables_and_foods": {
        "classes": [
            "Apple", "Artichoke", "Bagel", "Banana", "Bell pepper", "Bread",
            "Broccoli", "Burrito", "Cabbage", "Cake", "Candy", "Cantaloupe",
            "Carrot", "Cheese", "Cocktail", "Coconut", "Coffee (drink)",
            "Common fig", "Cookie", "Cream", "Croissant", "Cucumber", "Doughnut",
            "Egg", "French fries", "Garden Asparagus", "Grape", "Grapefruit",
            "Guacamole", "Hamburger", "Honeycomb", "Hot dog", "Ice cream",
            "Juice", "Lemon (plant)", "Mango", "Milk", "Muffin", "Mushroom",
            "Orange (fruit)", "Pancake", "Pasta", "Pastry", "Peach", "Pear",
            "Pineapple", "Pizza", "Pomegranate", "Popcorn", "Potato", "Pretzel",
            "Pumpkin", "Radish", "Salad", "Sandwich", "Squash (Plant)",
            "Strawberry", "Submarine sandwich", "Sushi", "Taco", "Tart",
            "Tea", "Tomato", "Waffle", "Watermelon", "Wine", "Winter melon", "Zucchini"
        ],
        "stable_attributes": ["color", "state", "shape", "placement"],
        "allowed_values": {
            "color": GLOBAL_ATTRIBUTE_VOCAB["color"],
            "state": ["whole/intact", "sliced/cut", "peeled", "bitten"],
            "shape": ["round", "triangular", "square", "rectangular"],
            "placement": ["on plate/bowl", "on table/cutting board", "in basket/box", "on tree/plant", "held in hand"]
        },
        # Đồ uống không có state whole/sliced -> bỏ state, thay placement phù hợp ly/chai
        "class_overrides": {
            "Cocktail": {"state": None, "placement": ["in cup/glass", "in bottle", "on table", "held in hand"]},
            "Coffee (drink)": {"state": None, "placement": ["in cup/glass", "in bottle", "on table", "held in hand"]},
            "Juice": {"state": None, "placement": ["in cup/glass", "in bottle", "on table", "held in hand"]},
            "Milk": {"state": None, "placement": ["in cup/glass", "in bottle", "on table", "held in hand"]},
            "Tea": {"state": None, "placement": ["in cup/glass", "in bottle", "on table", "held in hand"]},
            "Wine": {"state": None, "placement": ["in cup/glass", "in bottle", "on table", "held in hand"]}
        }
    },

    # NHÓM 16: Dụng cụ cầm tay, Dao kéo, Văn phòng phẩm, Phụ kiện nhỏ
    "tools_utensils_and_accessories": {
        "classes": [
            "Adhesive tape", "Axe", "Band-aid", "Binoculars", "Book",
            "Bottle opener", "Can opener", "Chainsaw", "Chisel",
            "Chopsticks", "Coin", "Cooking spray", "Crutch", "Cutting board",
            "Dagger", "Diaper", "Door handle", "Drill (Tool)", "Earring",
            "Envelope", "Eraser", "Face powder", "Facial tissue holder",
            "Fire hydrant", "Fork", "Grinder", "Hair spray", "Hammer", "Handgun",
            "Kitchen knife", "Knife", "Ladder", "Ladle", "Lipstick",
            "Nail (Construction)", "Necklace", "Paper cutter", "Paper towel",
            "Pen", "Pencil sharpener", "Perfume", "Pizza cutter",
            "Power plugs and sockets", "Pressure cooker", "Ratchet (Device)",
            "Rifle", "Ring binder", "Ruler", "Scissors", "Screwdriver",
            "Seat belt", "Shotgun", "Shower", "Sink", "Soap dispenser",
            "Spatula", "Spoon", "Stapler", "Stethoscope", "Stretcher",
            "Sword", "Syringe", "Tap", "Tire", "Toilet", "Toilet paper",
            "Toothbrush", "Wheel", "Whisk", "Wrench", "Bicycle wheel", "Beehive", "Cannon",
            "Bronze sculpture", "Bust", "Sculpture"
        ],
        "stable_attributes": ["color", "material", "state"],
        "allowed_values": {
            "color": GLOBAL_ATTRIBUTE_VOCAB["color"],
            "material": ["wooden", "metal", "plastic", "ceramic", "paper/cardboard", "stone"],
            "state": ["open", "closed", "folded"]
        },
        # Tượng thường bằng đá/kim loại, không có state open/closed
        "class_overrides": {
            "Bronze sculpture": {"state": None},
            "Bust": {"state": None},
            "Sculpture": {"state": None}
        }
    },

    # NHÓM 17: Thực vật & Hoa -> Tách khỏi nhóm dụng cụ: material/state không áp dụng,
    # chỉ color và placement là ổn định (in vase, chậu, trong vườn, trong nhà...)
    "plants_and_flowers": {
        "classes": [
            "Christmas tree", "Common sunflower", "Flower", "Houseplant",
            "Lavender (Plant)", "Lily", "Maple", "Palm tree", "Rose", "Willow"
        ],
        "stable_attributes": ["color", "placement"],
        "allowed_values": {
            "color": GLOBAL_ATTRIBUTE_VOCAB["color"],
            "placement": ["in vase/bouquet", "potted", "in garden", "indoors", "outdoors"]
        }
    }
}

# ==============================================================================
# 4. BLACKLIST CLASSES (CÁC CLASS CẦN LOẠI BỎ KHI TẠO QA VẬT THỂ NHỎ)
#    Bao gồm: Nhãn cha trừu tượng (Hypernyms), Bộ phận cơ thể người, Kiến trúc lớn
# ==============================================================================
BLACKLIST_CLASSES = {
    # 4.1. Nhãn cha quá chung chung (Hypernyms) -> Dễ gây nhập nhằng
    "Animal", "Auto part", "Baked goods", "Bathroom accessory", "Carnivore",
    "Cat furniture", "Clothing", "Cosmetics", "Dairy Product", "Dessert",
    "Drink", "Fashion accessory", "Fast food", "Food", "Footwear", "Fruit",
    "Furniture", "Hiking equipment", "Home appliance", "Insect", "Invertebrate",
    "Kitchen appliance", "Kitchen utensil", "Kitchenware", "Land vehicle",
    "Mammal", "Marine invertebrates", "Marine mammal", "Medical equipment",
    "Moths and butterflies", "Musical instrument", "Office supplies",
    "Personal care", "Plant", "Plumbing fixture", "Reptile", "Seafood",
    "Shellfish", "Snack", "Sports equipment", "Tableware", "Tool", "Toy",
    "Vegetable", "Vehicle", "Weapon",

    # 4.2. Bộ phận cơ thể người & Nhãn người chung (Dành riêng cho NWPU-Crowd, không trộn vào Mixed-object attribute)
    "Beard", "Boy", "Girl", "Human arm", "Human body", "Human ear",
    "Human eye", "Human face", "Human foot", "Human hair", "Human hand",
    "Human head", "Human leg", "Human mouth", "Human nose", "Man",
    "Person", "Woman", "Skull",

    # 4.3. Kiến trúc / Cảnh quan lớn (Không phù hợp làm vật thể nhỏ < 1%)
    "Bathtub", "Bidet", "Building", "Castle", "Convenience store",
    "Fireplace", "Fountain", "House", "Jacuzzi", "Lighthouse",
    "Office building", "Porch", "Skyscraper", "Stairs", "Swimming pool",
    "Tower", "Tree", "Tree house",

    # 4.4. Vũ khí quân sự (hiếm gặp trong ảnh đời thường, chủ yếu trong bối cảnh chiến sự)
    "Bomb", "Missile"
}


# ==============================================================================
# 5. HÀM TIỆN ÍCH TRA CỨU NHANH (HELPER FUNCTIONS CHO PIPELINE)
# ==============================================================================
def build_class_lookup() -> Dict[str, Dict[str, Any]]:
    """
    Tạo bảng tra cứu O(1) từ DisplayName -> Cấu hình Attribute của class đó.

    Áp dụng class_overrides của group lên cấu hình mặc định:
    - list  -> thay thế vocab của attribute đó.
    - None  -> bỏ attribute (không áp dụng được cho class này).
    - attribute mới (chưa có ở group) -> tự động thêm vào stable_attributes.
    """
    lookup: Dict[str, Dict[str, Any]] = {}
    for group_name, group_info in OIDV7_GROUP_SCHEMA.items():
        group_attrs = group_info["stable_attributes"]
        group_allowed = group_info["allowed_values"]
        overrides = group_info.get("class_overrides", {})
        for cls_name in group_info["classes"]:
            attrs = list(group_attrs)
            allowed = dict(group_allowed)
            for attr, values in overrides.get(cls_name, {}).items():
                if values is None:
                    allowed.pop(attr, None)
                    if attr in attrs:
                        attrs.remove(attr)
                else:
                    allowed[attr] = values
                    if attr not in attrs:
                        attrs.append(attr)
            lookup[cls_name] = {
                "group": group_name,
                "stable_attributes": attrs,
                "allowed_values": allowed
            }
    return lookup


CLASS_TO_ATTRIBUTE_CONFIG = build_class_lookup()


def is_valid_target_class(display_name: str) -> bool:
    """Kiểm tra xem một class trong CSV có phù hợp để tạo câu hỏi vật thể nhỏ không."""
    if display_name in BLACKLIST_CLASSES:
        return False
    return display_name in CLASS_TO_ATTRIBUTE_CONFIG


def get_class_attribute_Spec(display_name: str) -> Optional[Dict[str, Any]]:
    """
    Trả về danh sách các attribute ổn định và tập giá trị hợp lệ cho 1 DisplayName.
    Trả về None nếu class nằm trong Blacklist.
    """
    if not is_valid_target_class(display_name):
        return None
    return CLASS_TO_ATTRIBUTE_CONFIG[display_name]


def build_attribute_taxonomy() -> Dict[str, set]:
    """
    Gom toàn bộ giá trị allowed (GLOBAL + mọi group + class_overrides đã merge)
    thành taxonomy {attribute: set(values)} — dùng cho validate fallback
    khi không có spec theo class (vd: legacy data).
    """
    tax: Dict[str, set] = {cat: set(vals) for cat, vals in GLOBAL_ATTRIBUTE_VOCAB.items()}
    for cls_spec in CLASS_TO_ATTRIBUTE_CONFIG.values():
        for attr, vals in cls_spec["allowed_values"].items():
            tax.setdefault(attr, set()).update(vals)
    return tax


if __name__ == "__main__":
    # Self-check: đối chiếu schema với CSV gốc để phát hiện typo / class bị bỏ sót
    import csv
    from collections import Counter
    from pathlib import Path

    in_schema = [c for g in OIDV7_GROUP_SCHEMA.values() for c in g["classes"]]
    dup = {c: n for c, n in Counter(in_schema).items() if n > 1}
    assert not dup, f"Class trung lap giua cac group: {dup}"
    assert not (set(in_schema) & BLACKLIST_CLASSES), "Class vua nam trong group vua nam trong blacklist"

    # Moi stable attribute phai co allowed_values va khong chua stop-values
    for cls, spec in CLASS_TO_ATTRIBUTE_CONFIG.items():
        for attr in spec["stable_attributes"]:
            assert attr in spec["allowed_values"], f"{cls}: thieu allowed_values cho '{attr}'"
            bad = set(spec["allowed_values"][attr]) & STOP_ATTRIBUTE_VALUES
            assert not bad, f"{cls}.{attr} chua stop-values: {bad}"

    csv_path = Path(__file__).resolve().parent.parent / "data" / "metadata" / "oidv7-class-descriptions-boxable.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            names = {row[1].strip() for row in csv.reader(f) if len(row) > 1 and row[1] != "DisplayName"}
        typo = [c for c in in_schema if c not in names]
        assert not typo, f"Class khong khop CSV (typo?): {typo}"
        neither = sorted((names - set(in_schema)) - BLACKLIST_CLASSES)
        print(f"CSV: {len(names)} classes | trong group: {len(set(in_schema))} | "
              f"blacklist: {len(BLACKLIST_CLASSES & names)} | bo qua: {len(neither)}")
        for c in neither:
            print(f"  - {c}")
    print("Self-check OK")