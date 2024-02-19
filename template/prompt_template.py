import numpy as np
import random
## COT template
cot_template = {
    "csqa":{'Q': [
        "What do people use to absorb extra ink from a fountain pen?",
        "What home entertainment equipment requires cable?",
        "The fox walked from the city into the forest, what was it looking for?",
        "Sammy wanted to go to where the people were. Where might he go?",
        "Where do you put your grapes just before checking out?",
        "Google Maps and other highway and street GPS services have replaced what?",
        "Before getting a divorce, what did the wife feel who was doing all the work?"
    ],
    'choices': [
        ["(a) shirt pocket", "(b) calligrapher’s hand", "(c) inkwell", "(d) desk drawer", "(e) blotter"],
        ["(a) radio shack", "(b) substation", "(c) television", "(d) cabinet"],
        ["(a) pretty flowers", "(b) hen house", "(c) natural habitat", "(d) storybook"],
        ["(a) populated areas", "(b) race track", "(c) desert", "(d) apartment", "(e) roadblock"],
        ["(a) mouth", "(b) grocery cart", "(c) super market", "(d) fruit basket", "(e) fruit market"],
        ["(a) united states", "(b) mexico", "(c) countryside", "(d) atlas"],
        ["(a) harder", "(b) anguish", "(c) bitterness", "(d) tears", "(e) sadness"]
    ],
    'A': [
        "The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink. So the answer is (e)",
        "The answer must require cable. Of the above choices, only television requires cable. So the answer is (c)",
        "The answer must be something in the forest. Of the above choices, only natural habitat is in the forest. So the answer is (c)",
        "The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people. So the answer is (a)",
        "The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. So the answer is (b)",
        "The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. So the answer is (d)",
        "The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness. So the answer is (c)"
    ]},
    "strategyqa":{
    'Q': [
        "Do hamsters provide food for any animals?",
        "Could Brooke Shields succeed at University of Pennsylvania?",
        "Yes or no: Hydrogen’s atomic number squared exceeds number of Spice Girls?",
        "Yes or no: Is it common to see frost during some college commencements?",
        "Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?",
        "Yes or no: Would a pear sink in water?"
    ],
    'choices': [
        ["(a) no", "(b) yes"],
        ["(a) no", "(b) yes"],
        ["(a) no", "(b) yes"],
        ["(a) no", "(b) yes"],
        ["(a) no", "(b) yes"],
        ["(a) no", "(b) yes"]
    ],
    'A': [
        "Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So the answer is (b)",
        "Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. So the answer is (b)",
        "Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5. So the answer is (a)",
        "College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. So the answer is (b)",
        "The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. So the answer is (a)",
        "The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float. So the answer is (a)"
    ],
    'qd':[  # question decomposition
        'Q1. Are hamster considered prey?\nQ2. Are prey food for predators?\nA1.Hamsters are prey animals.\nA2.Prey are food for predators.\nThus, hamsters provide food for some animals. So the answer is (b)',
        'Q1. Which school did Brooke Shields go to?\nQ2. Is Princeton University a comparable university to University of Pennsylvania.\nA1.Brooke Shields went to Princeton University.\nA2.Princeton University is about as academically rigorous as the University of Pennsylvania.\nThus, Brooke Shields could also succeed at the University of Pennsylvania. So the answer is (b)',
        "Q1. What is the atomic number of hydrogen?\nQ2. What is the squared atomic number of hydrogen?\nQ3. How many spice girls are there?\nA1.Hydrogen has an atomic number of 1.\nA2. 1 squared is 1.\nA3. There are 5 Spice Girls.\nThus, Hydrogen’s atomic number squared is less than 5. So the answer is (a)",
        "Q1. When do college commencement ceremonies typically happen?\nQ2. Is December in winter?\nQ3. Can there be frost in winter?\nA1. College commencement ceremonies can happen in December, May, and June.\nA2. December is in the winter.\nA3. Yes, there can be frost in winter.\nThus, there could be frost at some commencements. So the answer is (b)",
        "Q1. How long was the War in Vietnam in 1945-46?\nQ2. What is the gestation period for a llama?\nA1. The War in Vietnam was 6 months.\nA2. The gestation period for a llama is 11 months.\nThus, a llama could not give birth twice during the War in Vietnam. So the answer is (a)",
        "Q1. What is the density of a pear?\nQ2. What is the density of water?\nA1. The density of a pear is about 0.6g/cm3.\nA2. The density of water is 1g/cm3.\nThus, a pear would float as the density of the pear is lower than water. So the answer is (a)"  
    ],
    'all_A':[
        ["Hamsters can be kept as domesticated pets. However pets could still be used as food.","Hamsters are prey animals. Prey are food for predators.","The most plausible reasoning is (b). So the answer is (b)"],
        ["Although Brooke Shields is primarily known for her acting and modeling career. This does not affect the fact that Princeton University is as good as University of Pennsylvania.", "Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania.", "The most plausible reasoning is (b). So the answer is (b)"],
        ["Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5.", "Hydrogen has an atomic number of 1. 1 squared is 1. Since there are five spice girls, Hydrogen’s atomic number squared would not exceed the number of Spice Girls.","The most plausible reasoning is (a). So the answer is (a)"],
        ["College commencements can happen in May and June are during warmer months but it could also happen in December which is winter and there can be frost.", "College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost.", "The most plausible reasoning is (b). So the answer is (b)"],
        ["The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam.", "The gestation period for a llama is 11 months, a llama can only give birth twice within the Vietnam's war if it last longer than 22 months.","The most plausible reasoning is (a). So the answer is (a)"],
        ["The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float.", "Water has a density of 1g/cm3, a pear would only sink in water if it has a density higher than water.","The most plausible reasoning is (a). So the answer is (a)"]
        ]
},
    "obqa":{
    'Q': [
        "The sun is responsible for what?",
        "When standing miles away from Mount Rushmore, what happens?",
        "When food is reduced in the stomach, what happens?",
        "Poison causes harm to which of the following?",
        "A magnet will stick to what?",
        "Deer are less safe in the woods because wolves have what?",
        "An electric car causes what?"
    ],
    'choices': [
        ["(a) Puppies learning new tricks", "(b) Children growing up and getting old", "(c) Flowers wilting in a vase", "(d) Plants sprouting, blooming and wilting"],
        ["(a) The mountains seem very close", "(b) The mountains are boring", "(c) The mountains look the same as from up close", "(d) The mountains seem smaller than in photographs"],
        ["(a) The mind needs time to digest", "(b) Take a second to digest what I said", "(c) Nutrients are being deconstructed", "(d) Reader’s digest is a body of works"],
        ["(a) A Tree", "(b) A robot", "(c) A house", "(d) A car"],
        ["(a) A belt buckle", "(b) A wooden table", "(c) A plastic cup", "(d) A paper plate"],
        ["(a) Have fur", "(b) Howl", "(c) Have claws", "(d) Have tails"],
        ["(a) More CO2 emissions", "(b) Equal CO2 emissions", "(c) Electric emissions", "(d) Less CO2 emissions"]
    ],
    'A': [
        "A plant requires sunlight for photosynthesis, which accumulates resources required for sprouting, blooming and wilting. So the answer is (d)",
        "When an object is far away, it takes up less of your field of view, and so seems smaller than in the photographs. So the answer is (d)",
        "The stomach is part of the digestive system. The breaking down of food into nutrients occurs in the digestive system. So the answer is (c)",
        "A tree is a living thing. Poison causes harm to living things. So the answer is (a)",
        "A belt buckle is made of metal. If a magnet is attracted to a metal then that magnet will stick to that metal. So the answer is (a)",
        "Claws are used by wolves to catch prey like deer. So the answer is (c)",
        "An electric car uses less gasoline than a regular car and thus causes less CO2 emissions. So the answer is (d)"
    ],
    'qd': [
        "Q1. What do the sun provide??\nQ2. What can sunlight be used for?\nA1. The sun provides sunlight.\nA2. Sunlight can be used for photosynthesis which accumulates resources required for sprouting, blooming, and wilting of plants.\nThus, sunlight is responsible for plants sprouting, blooming, and wilting. So the answer is (d)",
        "Q1. What happens to the perception of an object when it's far away?\nQ2. How does an object's size in the field of view relate to its perceived size?\nA1. When an object is far away, it takes up less of your field of view.\nA2. An object seems smaller when it takes up less of your field of view than in photographs.\nThus, when standing miles away from Mount Rushmore, the mountains seem smaller than in photographs. So the answer is (d)",
        "Q1. Where does the breaking down of food into nutrients occur?\nQ2. What is the role of the stomach in digestion?\nA1. The breaking down of food into nutrients occurs in the digestive system.\nA2. The stomach is part of the digestive system.\nThus, when food is reduced in the stomach, nutrients are being deconstructed. So the answer is (c)",
        "Q1. What does poison harm?\nQ2. Which of the choices are living things?\nA1. Poison causes harm to living things.\nA2. A tree is a living thing.\nThus, poison causes harm to a tree. So the answer is (a)",
        "Q1. What materials are magnets attracted to?\nQ2. Which of the choices are made of metal?\nA1. Magnets are attracted to metal.\nA2. Belt buckle is made of metal.\nThus, a magnet will stick to a belt buckle. So the answer is (a)",
        "Q1. What do wolves use to catch prey?\nQ2. Is a deer a prey for wolves?\nA1. Claws are used by wolves to catch prey.\nA2. Deer are prey for wolves.\nThus, deer are less safe in the woods because wolves have claws. So the answer is (c)",
        "Q1. How does an electric car's gasoline usage compare to a regular car?\nQ2. How does gasoline usage relate to CO2 emissions?\nA1. An electric car uses less gasoline than a regular car.\nA2. Using less gasoline causes less CO2 emissions.\nThus, an electric car causes less CO2 emissions. So the answer is (d)"
],
    'all_A':[
        ["Puppies learn tricks regardless of the sun.", "The aging process is independent of the sun.", "The sun doesn't cause flowers in a vase to wilt; other factors like water availability do.", "A plant requires sunlight for photosynthesis, which accumulates resources required for sprouting, blooming and wilting.", "The most plausible reasoning is (d). So the answer is (d)"],
        
        [
        "Being far from an object doesn't change its inherent qualities, like being boring or exciting.",
        "The visual perception of mountains changes based on distance, so they won't look the same from miles away as from up close.",
        "Mountains seem smaller when viewed from a distance because they take up less of the viewer's field of view.",
        "When an object is far away, it takes up less of your field of view, and so seems smaller than in photographs.",
        "The most plausible reasoning is (d). So the answer is (d)."
        ],
        [
            "Digestion is a physiological process, and 'mind needing time to digest' is just a metaphor.",
            "The phrase 'Take a second to digest what I said' is metaphorical and not related to the stomach's function.",
            "The stomach is part of the digestive system. The breaking down of food into nutrients occurs in the digestive system.",
            "Reader’s Digest is a publication, and it has no relation to the digestive process in the stomach.",
            "The most plausible reasoning is (c). So the answer is (c)."
        ],
        [
            "A tree is a living thing. Poison causes harm to living things.",
            "Robots are machines and cannot be poisoned.",
            "Houses are inanimate structures and are not affected by poisons in the way living organisms are.",
            "Cars, being mechanical devices, cannot be poisoned in the traditional sense.",
            "The most plausible reasoning is (a). So the answer is (a)."
        ],
        [
            "A belt buckle is made of metal. If a magnet is attracted to a metal then that magnet will stick to that metal.",
            "Wood is not a ferromagnetic material and cannot be attracted to magnets.",
            "Plastic is not a magnetic material, and magnets are not attracted to it.",
            "Paper is not magnetic, so magnets won't stick to it.",
            "The most plausible reasoning is (a). So the answer is (a)."
        ],
        [
            "Deer's safety is not threatened by the fur of wolves.",
            "Wolves' howling is a form of communication and does not directly affect the safety of deer.",
            "Claws are used by wolves to catch prey like deer.",
            "The presence of tails in wolves does not impact the safety of deer in any way.",
            "The most plausible reasoning is (c). So the answer is (c)."
        ],
        [
            "Electric cars do not run on gasoline, so they don't produce CO2 emissions from burning fuel.",
            "Electric cars, when charged from clean energy sources, can have zero emissions during their operation.",
            "'Electric emissions' is not a standard term in the context of car emissions.",
            "An electric car uses less gasoline than a regular car and thus causes less CO2 emissions.",
            "The most plausible reasoning is (d). So the answer is (d)."
        ]
        ]
    
},
    "qasc":{ 
    'Q': [
        "How do you reduce pollution?",
        "What will move to another area if their habitat will no longer support them?",
        "With the exception of allergies, what may cause a person to seek medical attention?",
        "Lavender can induce what?",
        "What state is a liquid in when frozen?",
        "What unites to form a diploid zygote?",
        "What absorbs all visible light?"
    ],
    'choices': [
        ["(a) Igniting fuel and oxidiser", "(b) Transportation technology", "(c) Wasting", "(d) Not recycling", "(e) Burning fossil fuels", "(f) Converting electricity to heat", "(g) Water conservation", "(h) Using less resources"],
        ["(a) Density", "(b) Birds", "(c) Squids", "(d) Humans", "(e) Clouds", "(f) Gravity", "(g) Cows", "(h) Whales"],
        ["(a) Contact with latex", "(b) A tree falling", "(c) Organs within the body", "(d) Contact with baby chicks", "(e) Prolactin release", "(f) Contact with peanut butter", "(g) Hypothyroidism", "(h) Contact with microorganisms"],
        ["(a) Healing", "(b) Energy", "(c) Hormones", "(d) Mutations", "(e) Heart rate", "(f) Growth", "(g) Symptoms", "(h) Warmth"],
        ["(a) Vapor", "(b) Dense", "(c) Gas", "(d) Cooled", "(e) Steam", "(f) Solid", "(g) Boiling", "(h) Cold"],
        ["(a) Plant reproduction", "(b) Most plants", "(c) Orchids", "(d) Sperm and ova", "(e) Salt and pepper", "(f) Predator and prey", "(g) Honeybees", "(h) Diploids and zygotes"],
        ["(a) Apples", "(b) Coal", "(c) Green", "(d) Coral", "(e) Skin", "(f) Bamboo", "(g) Glass", "(h) Eyes"]
    ],
    'A': [
        "Conserving resources has a positive impact on the environment. Use of resources affects the environment such as pollution. So the answer is (h)",
        "If a habitat can no longer support animals then those animals will move to another area. Cows are social animals. So the answer is (g)",
        "Microorganisms can cause infections. Infections usually require medical treatment. So the answer is (h)",
        "Healing requires rest. Lavender induces restful sleep. So the answer is (a)",
        "Freezing means changing from a liquid into a solid by reducing heat energy. Liquids freeze when they change to the solid state. So the answer is (f)",
        "Gametes then unite in fertilization and form a diploid zygote. Collectively, the sperm and the ova are also referred to as gametes. So the answer is (d)",
        "If an object is black then that object absorbs all visible light. Light grains are quartz, Black grains are coal. So the answer is (b)"
    ],
    'qd': [
        "Q1. What has a positive impact on the environment?\nQ2. How does the use of resources affect the environment?\nA1. Conserving resources has a positive impact on the environment.\nA2. Use of resources affects the environment such as pollution.\nThus, to reduce pollution, using less resources is beneficial. So the answer is (h)",
        "Q1. What happens to animals when their habitat can no longer support them?\nQ2. Which of the choices are social animals?\nA1. If a habitat can no longer support animals then those animals will move to another area.\nA2. Cows are social animals.\nThus, cows will move to another area if their habitat will no longer support them. So the answer is (g)",
        "Q1. What can cause infections?\nQ2. What usually requires medical treatment?\nA1. Microorganisms can cause infections.\nA2. Infections usually require medical treatment.\nThus, with the exception of allergies, contact with microorganisms may cause a person to seek medical attention. So the answer is (h)",
        "Q1. What does healing require?\nQ2. What does lavender induce?\nA1. Healing requires rest.\nA2. Lavender induces restful sleep.\nThus, lavender can induce healing. So the answer is (a)",
        "Q1. What does freezing mean in terms of state change?\nQ2. What happens to liquids when they freeze?\nA1. Freezing means changing from a liquid into a solid by reducing heat energy.\nA2. Liquids freeze when they change to the solid state.\nThus, when a liquid is frozen, it is in a solid state. So the answer is (f)",
        "Q1. What unites in fertilization to form a zygote?\nQ2. Which of the choices are referred to as gametes?\nA1. Gametes then unite in fertilization and form a diploid zygote.\nA2. Collectively, the sperm and the ova are also referred to as gametes.\nThus, sperm and ova unite to form a diploid zygote. So the answer is (d)",
        "Q1. What does an object do if it is black in terms of light?\nQ2. What are black grains?\nA1. If an object is black then that object absorbs all visible light.\nA2. Light grains are quartz, Black grains are coal.\nThus, coal absorbs all visible light. So the answer is (b)"
],
    'all_A':[ # longer context.
        [
        "Igniting fuel and oxidizer typically produces emissions, contributing to pollution.",
        "Transportation technology, especially if not sustainable, can contribute to pollution.",
        "Wasting, especially of natural resources, can exacerbate pollution.",
        "Not recycling can lead to increased waste and can contribute to pollution.",
        "Burning fossil fuels is a major source of pollution, especially CO2 emissions.",
        "Converting electricity to heat does not directly correlate with reducing pollution.",
        "Water conservation is important for preserving water resources but doesn't directly reduce air pollution.",
        "Conserving resources has a positive impact on the environment. Use of resources affects the environment such as pollution.",
        "The most plausible reasoning is (h). So the answer is (h)."
    ],
    [
        "Density is a measure of mass per volume and doesn't move due to habitat changes.",
        "Birds migrate based on various factors, one of which could be habitat changes.",
        "Squids are marine animals; if their marine habitat changes, they might move, but not necessarily to 'another area'.",
        "Humans can migrate due to various reasons including environmental ones.",
        "Clouds move based on atmospheric conditions, not due to habitat.",
        "Gravity is a force and doesn't move based on habitat changes.",
        "If a habitat can no longer support animals then those animals will move to another area. Cows are social animals.",
        "Whales, being marine mammals, could change locations if their habitat is threatened, but they're not known to migrate on land.",
        "The most plausible reasoning is (g). So the answer is (g)."
    ],
    [
        "Contact with latex can cause allergic reactions in some people.",
        "A tree falling is a physical hazard and can cause injuries. But it does not state if it falls on a human.",
        "Organs within the body don't typically cause a person to seek medical attention unless there's a dysfunction.",
        "Contact with baby chicks can expose one to diseases like Salmonella.",
        "Prolactin release is associated with breastfeeding and doesn't typically cause someone to seek medical attention.",
        "Contact with peanut butter can cause allergic reactions in some individuals.",
        "Hypothyroidism is a medical condition that affects the thyroid gland and may require medical attention.",
        "Microorganisms can cause infections. Infections usually require medical treatment.",
        "The most plausible reasoning is (h). So the answer is (h)."
    ],
    [
        "Healing requires rest. Lavender induces restful sleep.",
        "Lavender does not directly induce energy.",
        "Lavender doesn't induce the production of hormones.",
        "Lavender does not cause genetic mutations.",
        "Lavender does not directly influence heart rate.",
        "Lavender does not induce growth.",
        "Lavender can alleviate certain symptoms, but it doesn't directly 'induce' them.",
        "Lavender has a cooling effect, not necessarily warmth.",
        "The most plausible reasoning is (a). So the answer is (a)."
    ],
    [
        "Vapor is a gaseous state, not a frozen state.",
        "Dense refers to the compactness of matter, not its state.",
        "Gas is not the state of a liquid when frozen.",
        "Cooled just means reduced in temperature but doesn't specify a state.",
        "Steam is a gaseous state of water.",
        "Freezing means changing from a liquid into a solid by reducing heat energy. Liquids freeze when they change to the solid state.",
        "Boiling is a process that turns a liquid into a gas.",
        "Cold is a relative term and doesn't define a state of matter.",
        "The most plausible reasoning is (f). So the answer is (f)."
    ],
    [
        "Plant reproduction is a broad term and doesn't specify which entities unite.",
        "Most plants reproduce through pollen and ovules, but the term 'most plants' is too general.",
        "Orchids have unique reproductive structures but they don't form diploid zygotes by themselves.",
        "Gametes then unite in fertilization and form a diploid zygote. Collectively, the sperm and the ova are also referred to as gametes.",
        "Salt and pepper are condiments and don't unite to form zygotes.",
        "Predator and prey represent ecological interactions, not reproductive ones.",
        "Honeybees play a role in plant pollination but don't unite to form zygotes.",
        "Diploids are cells with two sets of chromosomes, and they don't unite with zygotes.",
        "The most plausible reasoning is (d). So the answer is (d)."
    ],
    [
        "Apples reflect light in the visible spectrum and do not absorb all visible light.",
        "If an object is black then that object absorbs all visible light. Light grains are quartz, Black grains are coal.",
        "Green objects reflect green light and absorb other colors in the visible spectrum.",
        "Corals come in various colors and do not absorb all visible light.",
        "Skin does not absorb all visible light; its color depends on various factors including melanin.",
        "Bamboo is typically green or light brown and does not absorb all visible light.",
        "Glass is transparent and does not absorb all visible light.",
        "Eyes reflect and absorb certain wavelengths of light but do not absorb all visible light.",
        "The most plausible reasoning is (b). So the answer is (b)."
    ]
    ]
}
}

sys_prompt = "You are a helpful, respectful and honest assistant. You are to answer the last question by first providing a reasoning followed by the answer, similar to previous examples.\n"

fs_prompt_ans = {'cot_qd':'qd','cot_cf': 'all_A'} # set to different answer


def format_llama_prompt(context,choices=None,fs_prompt=None,choice_joined=False,prompt_type= 'cot',explanation = None,num_shot = 3):
    """
    Shuffle and take num_shot number of shots from the fs_prompt
    Configure template according to prompt_type
    """
    if fs_prompt is not None:
        starting_prompt = []
        fs_prompt_a = fs_prompt_ans.get(prompt_type,'A')
        random_ids = np.arange(len(fs_prompt['Q']))
        # if num_shot < len(fs_prompt['Q']):
        #     np.random.shuffle(random_ids)
        max_shot_ids = random_ids[:num_shot].tolist()
        fs_q = [fs_prompt['Q'][i] for i in max_shot_ids]
        fs_c = [fs_prompt['choices'][i] for i in max_shot_ids]
        fs_a = [fs_prompt[fs_prompt_a][i] for i in max_shot_ids]
        
        for i,(q,c,a) in enumerate(zip(fs_q,fs_c,fs_a)):
            curr_p = []
            curr_p.append(f"Q: {q}")
            curr_p.append('Answer choices:\n'+'\n'.join(c))
            if prompt_type in['cot' ,'cot_llama','cot_sc','cot_refine','cot_sec']:
                a_text = f"A: {a}"
            elif prompt_type == 'cot_sbs':# sbs means step by step
                a_text = f"A: Let's think step by step. {a}"
            elif prompt_type == 'no_cot':
                a_only = a.split('So the answer is')[-1]
                a_text = 'So the answer is ' + a_only.strip()
            elif prompt_type == 'cot_qd': # question decomposition
                a_text  = f"Lets break down the problem.\n{a}"
            elif prompt_type == 'cot_cf': # counterfactual reasoning
                a_text = ["Lets think about all the given choices."]
                for j,sub_a in enumerate(a[:-1]):
                    a_text += [f"Reasoning ({chr(ord('a') + j)}) {sub_a}"]
                    # a_text += [f"C{j+1}. {sub_a}"] # prepend a C{digit}
                a_text += [a[-1]]
                a_text = '\n'.join(a_text) 
            curr_p.append(a_text)
            starting_prompt.append('\n'.join(curr_p))

        starting_prompt = '\n\n'.join(starting_prompt) + '\n\n'
    else:
        starting_prompt = ''
    ## To add answer choices
    if choices is not None:
        if not choice_joined:
            num_to_alpha =  {str(i):chr(ord('a') + i) for i in range(len(choices))}
            starting_s = ["Answer choices:"]
            for i,choice in enumerate(choices):
                starting_s.append(f"({num_to_alpha[str(i)]}) {choice}")
            added_prompt = '\n'.join(starting_s)
        else:
            if '"' in choices:
                choices = choices.replace('"','')
            split_choice = [c.strip() for c in choices.split(',')]
            for i,choice in enumerate(split_choice):
                split_choice[i] = f"({chr(ord('a') + i)}) {choice}"
            split_choice = ["Answer choices:"] + split_choice
            added_prompt = '\n'.join(split_choice) 
    else:
        added_prompt = ""
    

    out_prompt = starting_prompt + 'Q: '+ context + '\n' + added_prompt + '\nA:' 
    out_prompt += prompt_input(prompt_type,explanation)
    if prompt_type == 'cot_llama':
        out_prompt = f'[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n{out_prompt}[/INST]' # using llama prompt
    
    return out_prompt


def prompt_input(prompt_type,explanation=None,add_postfix=True):
    out_prompt = ''
    if prompt_type == 'cot_sbs':
        out_prompt += " Let's think step by step."
    elif prompt_type == 'cot_qd':
        out_prompt += " Let's break down the problem."
    elif prompt_type == 'cot_cf':
        out_prompt += " Let's think about all the given choices."
    if explanation is not None: # used during perturbation
        if prompt_type  == 'cot_qd':
            explanation = '\n'.join(explanation)
        elif prompt_type == 'cot_cf':
            add_postfix = False
            cf_expl =[]
            for k,sub_expl in enumerate(explanation):
                cf_expl += [f"Reasoning ({chr(ord('a') + k)}) {sub_expl}"]
            explanation = '\n'.join(cf_expl)
        elif prompt_type != 'no_cot':
            if explanation[-1] != '.':
                explanation += '.'
        if prompt_type not in ['cot','cot_sc','cot_refine','cot_sec']:
            explanation = '\n' + explanation.strip()
            
        out_prompt += explanation
        
        if add_postfix:
            out_prompt += ' So the answer is '
    return out_prompt


ent_prompt = "Given a premise and hypothesis, predict if the hypothesis entails the premise.\n Premise: {p}\n Hypothesis: {h}"
ent_fs = {'p': ['This church choir sings to the masses as they sing joyous songs from the book at a church.',
                  'This church choir sings to the masses as they sing joyous songs from the book at a church.',
                  'A woman with a green headscarf, blue shirt and a very big grin.',
                  'An old man with a package poses in front of an advertisement.',
                  'A statue at a museum that no seems to be looking at.',
                  'A land rover is being driven across a river.',
                  'A man playing an electric guitar on stage.'
                  ],
            'h':['The church is filled with song.',
                 'A choir singing at a baseball game.',
                 'The woman is very happy.',
                 'A man walks by an ad.',
                 'The statue is offensive and people are mad that it is on display.',
                 'A Land Rover is splashing water as it crosses a river.',
                 'A man playing guitar on stage.'
                 ],
            'a':['yes',
                 'no',
                 'yes',
                 'no',
                 'no',
                 'yes',
                 'yes'
                 ]}

def create_entailment_prompt(q,yhat,expl):
    ans_prompt  = "A:{a}"
    fs_all = []
    for fs_i,fs_p in enumerate(ent_fs['p']):
        curr_fs = []
        fs_inst = ent_prompt.format_map({'p':fs_p,'h':ent_fs['h'][fs_i]})
        curr_fs.append(fs_inst)
        curr_fs.append(ans_prompt.format_map({'a':ent_fs['a'][fs_i]}))
        fs_all.append('\n'.join(curr_fs))
        
    fs_all = '\n\n'.join(fs_all)
    q  = q = ' ' + yhat.strip()
    input_ = ent_prompt.format_map({'p':q,'h':expl})
    out = fs_all + '\n\n' + input_ + "A:"
    return out

