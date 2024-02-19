fs_shot = {
    'add_mistakes':[{'rationale':"requires sunlight for photosynthesis",
                    'explanation':"A plant requires sunlight for photosynthesis, which accumulates resources required for sprouting, blooming and wilting.",
                    "output":"A plant requires money for survival, which accumulates resources required for opening a club",
                    'answer':"Plants sprouting, blooming and wilting",
                    },
                    {'rationale':"breaking down of food into nutrients",
                    'explanation':"The stomach is part of the digestive system. The breaking down of food into nutrients occurs in the digestive system.",
                    "output":"The stomach is part of the digestive system. The watering of seeds into plants occurs in the digestive system.",
                    'answer':"Nutrients are being deconstructed",
                    },
                    {'rationale':"a living thing",
                    'explanation':"A tree is a living thing. Poison causes harm to living things.",
                    "output":"A stone wall is already dead. Poison causes harm to living things.",
                    'answer':"A Tree",}
                    ],
           'rationale_extraction':[
                            {'question':"The sun is responsible for,",
                            'answer':"plants sprouting, blooming and wilting",
                            'explanation':"A plant requires sunlight for photosynthesis, which accumulates resources required for sprouting, blooming and wilting.",
                            "output":"requires sunlight for photosynthesis"},
                           {'question':"When food is reduced in the stomach,",
                            'answer':"nutrients are being deconstructed",
                            'explanation':"The stomach is part of the digestive system. The breaking down of food into nutrients occurs in the digestive system.",
                            "output":"breaking down of food into nutrients"},
                           {'question':"Poison causes harm to which of the following?",
                            'answer':"A Tree",
                            'explanation':"A tree is a living thing. Poison causes harm to living things.",
                            "output":"a living thing"},
                           ],
           'cf':{'obqa':
                [{
                        'question':"The sun is responsible for,",
                        'choices':["puppies learning new tricks", "children growing up and getting old", "flowers wilting in a vase", "plants sprouting, blooming and wilting"],
                        'answer': "(d) plants sprouting, blooming and wilting",
                        'output':"The dry weather is responsible for,",
                        'target': "sun",
                        'edit': "dry weather",
                        'target_answer': "(c) flowers wilting in a vase"
                    },
                    {
                        'question':"Which of these things will supplement a squirrel's caloric needs?",
                        'choices':["fish", "tree bark", "nuts", "leaves"],
                        'answer': "(c) nuts",
                        'output':"Which of these things will supplement a bear's caloric needs?",
                        'target': "squirrel",
                        'edit': "bear",
                        'target_answer': "(a) fish"
                        
                    },
                    {
                        'question':"When standing miles away from Mount Rushmore,",
                        'choices':["the mountains seem very close", "the mountains are boring", "the mountains look the same as from up close", "the mountains seem smaller than in photographs"],
                        'answer': "(d) the mountains seem smaller than in photographs",
                        'output':"When standing near to Mount Rushmore,",
                        'target': "miles away from",
                        'edit': "near to",
                        'target_answer': "(a) the mountains seem very close"
                    },
                    ],
           'qasc':[{
                        'question':"Removing what from food will preserve it?",
                        'choices':["flavor", "body water", "heat energy", "color", "Water", "Bodily water", "moisture", "ingredients"],
                        'answer': "(g) moisture",
                        'output':"Removing what from food will cool it down?",
                        'target': "preserve it",
                        'edit': "cool it down",
                        'target_answer': "(c) heat energy"
                    },
                    {
                        'question':"Which of the following  has the most antioxidant benefits for the body?",
                        'choices':["preserved muskrat", "preserved blueberries", "antibiotics", "hamburger", "hydrogen peroxide", "prolactin release", "evaporation", "Thyroid-stimulating hormone"],
                        'answer': "(b) preserved blueberries",
                        'output':"Which of the following  has the worst health benefits for the body?",
                        'target': "most antioxidant",
                        'edit': "worst health",
                        'target_answer': "(d) hamburger"
                        
                    },
                    {
                        'question':"What is the process by which living things give rise to offspring?",
                        'choices':["sex", "diploid", "ovum", "bird", "ovary", "eggs", "gametes", "DNA"],
                        'answer': "(a) sex",
                        'output':"What chickens lay to give rise to offspring?",
                        'target': "is the process by which living things",
                        'edit': "chickens lay to",
                        'target_answer': "(f) eggs"
                    }],
           },
        'cf_edit': [
                    {
                        'question':"The sun is responsible for,",
                        'cf_question':"The dry weather is responsible for,",
                        'original': "sun",
                        'changed': "dry weather"
                    },
                    {
                        'question':"Someone wants their electromagnets to work, but is having difficulty powering them. In order to make them work, they need to,",
                        'cf_question':"Someone wants to install their electromagnets, but is having difficulty. In order to install them, they need to,",
                        'original': "their electromagnets to work,powering them.,make them work",
                        'changed': "to install their electromagnets,install them"
                    },
                    {
                        'question':"In the hottest months in the hottest desert, creatures such as birds may find water to drink,",
                        'cf_question':"In the hottest months on the beach, where may creatures such as birds find water to drink?",
                        'original': "hottest dessert",
                        'changed': "beach"
                    }
        ]
            }
fs_shot_qd = {
    'add_mistakes':[
                    {'question':"What do the sun provide?",
                    'answer':"The sun provides sunlight.",
                    'rationale':"sunlight",
                    'output':"The sun provides water"
                    },
                    {'question':"Where does the breaking down of food into nutrients occur?",
                    'answer':"The breaking down of food into nutrients occurs in the digestive system.",
                    'rationale':"digestive system.",
                    'output':"The breaking down of food into nutrients occurs in the nervous system"
                    },
                    {'question':"What does poison harm?",
                    'answer':"Poison causes harm to living things.",
                    "rationale":"living things.",
                    "output":"Poison causes harm to happy things"}
                    ],
    'rationale_extraction':[
                    {'question':"What do the sun provide?",
                    'answer':"The sun provides sunlight.",
                    'output':"sunlight"
                    },
                    {'question':"Where does the breaking down of food into nutrients occur?",
                    'answer':"The breaking down of food into nutrients occurs in the digestive system.",
                    'output':"digestive system."
                    },
                    {'question':"What does poison harm?",
                    'answer':"Poison causes harm to living things.",
                    "output":"living things."},
                    ],
}