from template.prompt_template import cot_template
import re
import numpy as np
refine_template = {
    'strategyqa':{
        'init': [
            "Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So the answer is (b)",
            "Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. So the answer is (b)",
            "Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5. So the answer is (a)",
            "College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. So the answer is (b)",
            "The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. So the answer is (a)",
            "The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float. So the answer is (a)"
        ],
        'feedback': [
            {'sentence': "Hamsters are rodent animals. Rodents are predators. So the answer is (a).",'feedback': "Factuality: Sentence is not factual. 1/5\nRelevance: Sentence has some relevance to the question. 3/5\nInformativeness: The sentence does not provide much information for producing the answer. 2/5\nTotal Score: 8/15"},
            {
            'sentence': "Brooke Shields attended a university. Some people believe celebrities can't study well. So the answer is (a).",
            'feedback': "Factuality: Sentence is somewhat factual. 3/5\nRelevance: Sentence is moderately related to the question. 3/5\nInformativeness: The sentence provides a weak generalization. 2/5\nTotal Score: 8/15"
            },
            {
            'sentence': "Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5. So the answer is (a)",
            'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides direct reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
            },
            {
            'sentence': "Some commencements happen in spring. Spring can be chilly but not frosty. So the answer is (a).",
            'feedback': "Factuality: While spring can be chilly, it isn't the only time for commencements. 3/5\nRelevance: Sentence is related to the time of commencements. 4/5\nInformativeness: The sentence doesn't consider other commencement times. 3/5\nTotal Score: 10/15"
            },
            {
            'sentence': "The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. So the answer is (a)",
            'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides exact reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
            },
            {
            'sentence': "The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float. So the answer is (a)",
            'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides direct and accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
        }
        ],
        'refine':[
        [{'sentence': "Hamsters are rodent animals. Rodents are predators. So the answer is (a).",'feedback': "Factuality: Sentence is not factual. 1/5\nRelevance: Sentence has some relevance to the question. 3/5\nInformativeness: The sentence does not provide much information for producing the answer. 2/5\nTotal Score: 8/15"},
        {'sentence': "Hamsters are rodents. Rodents are preys for snakes. So the answer is (b).","feedback": "Factuality: Sentence is somewhat factual. 3/5\nRelevance: Sentence is relevant to the question. 4/5\nInformativeness: The sentence has some ambiguity in answering the question. 3/5\nTotal Score: 10/15"},
        {'sentence': "Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So the answer is (b)",'feedback': "Factuality: Sentence is factual. 4/5\nRelevance: Sentence is relevant to the question. 5/5\nInformativeness: The sentence provides the correct reasoning to the answer. 5/5\nTotal Score: 14/15. Stop refining the answer."}
        ],
    [
        {
            'sentence': "Brooke Shields has acted in movies. Movies are unrelated to universities. So the answer is (a).",
            'feedback': "Factuality: Sentence is factual but unrelated. 2/5\nRelevance: Sentence is not relevant to the question. 2/5\nInformativeness: The sentence does not provide useful information for answering. 2/5\nTotal Score: 6/15"
        },
        {
            'sentence': "Brooke Shields attended a university. Some people believe celebrities can't study well. So the answer is (a).",
            'feedback': "Factuality: Sentence is somewhat factual. 3/5\nRelevance: Sentence is moderately related to the question. 3/5\nInformativeness: The sentence provides a weak generalization. 2/5\nTotal Score: 8/15"
        },
        {
            'sentence': "Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. So the answer is (b)",
            'feedback': "Factuality: Sentence is factual. 4/5\nRelevance: Sentence is very relevant to the question. 5/5\nInformativeness: The sentence provides direct reasoning. 5/5\nTotal Score: 14/15. Stop refining the answer."
        }
    ],
    [
        {
            'sentence': "Hydrogen is the lightest element. Light elements have bigger atomic numbers than heavy ones. So the answer is (b).",
            'feedback': "Factuality: Sentence is incorrect. 1/5\nRelevance: Sentence is related to hydrogen but not to the atomic number squared and no mention of spice girls. 3/5\nInformativeness: The sentence misguides the answer. 2/5\nTotal Score: 6/15"
        },
        {
            'sentence': "Hydrogen's atomic number is low. When squared, it might be close to the number of Spice Girls. So the answer is (a).",
            'feedback': "Factuality: Sentence is vague but correct. 3/5\nRelevance: Sentence is relevant to the question. 4/5\nInformativeness: The sentence is ambiguous about the exact number. 3/5\nTotal Score: 10/15"
        },
        {
            'sentence': "Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5. So the answer is (a)",
            'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides direct reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
        }
    ],

    [
        {
            'sentence': "Colleges often have indoor commencements. Inside buildings, there isn't frost. So the answer is (a).",
            'feedback': "Factuality: While some commencements are indoors, not all are. 2/5\nRelevance: It has partial relevance, focusing on indoor commencements. 3/5\nInformativeness: It omits the possibility of outdoor commencements. 2/5\nTotal Score: 7/15"
        },
        {
            'sentence': "Some commencements happen in spring. Spring can be chilly but not frosty. So the answer is (a).",
            'feedback': "Factuality: While spring can be chilly, it isn't the only time for commencements. 3/5\nRelevance: Sentence is related to the time of commencements. 4/5\nInformativeness: The sentence doesn't consider other commencement times. 3/5\nTotal Score: 10/15"
        },
        {
            'sentence': "College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. So the answer is (b)",
            'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides a comprehensive reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
        }
    ],

    [
        {
            'sentence': "Llamas are animals. The Vietnam War was a long event. So the answer is (b).",
            'feedback': "Factuality: The sentence is simplistic and non-specific. 1/5\nRelevance: The sentence doesn't connect llamas' gestation to the war duration. 2/5\nInformativeness: Provides no useful information for the answer. 1/5\nTotal Score: 4/15"
        },
        {
            'sentence': "Llamas have long pregnancies. The Vietnam War lasted for years. So the answer is (b).",
            'feedback': "Factuality: The Vietnam War's mentioned duration is incorrect. 2/5\nRelevance: Tries to correlate gestation and war duration but is inaccurate. 3/5\nInformativeness: Provides a flawed reasoning for the answer. 2/5\nTotal Score: 7/15"
        },
        {
            'sentence': "The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. So the answer is (a)",
            'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides exact reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
        }
    ],

    [
        {
            'sentence': "Pears are fruits. Most fruits sink in water. So the answer is (b).",
            'feedback': "Factuality: The assumption about fruits is incorrect. 2/5\nRelevance: The sentence relates pears with fruits and water but in a flawed manner. 3/5\nInformativeness: Provides a misguided reasoning. 2/5\nTotal Score: 7/15"
        },
        {
            'sentence': "Pears are slightly heavy. Heavy things might sink in water. So the answer is (b).",
            'feedback': "Factuality: The assumption about pears' weight is not comprehensive. 3/5\nRelevance: Tries to relate weight and sinking but is simplistic. 3/5\nInformativeness: Provides a vague reasoning for the answer. 3/5\nTotal Score: 9/15"
        },
        {
            'sentence': "The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float. So the answer is (a)",
            'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides direct and accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
        }
    ]
        ]
    },
    'obqa':{
        'init': [
        "A plant requires sunlight for photosynthesis, which accumulates resources required for sprouting, blooming and wilting. So the answer is (d)",
        "When an object is far away, it takes up less of your field of view, and so seems smaller than in the photographs. So the answer is (d)",
        "The stomach is part of the digestive system. The breaking down of food into nutrients occurs in the digestive system. So the answer is (c)",
        "A tree is a living thing. Poison causes harm to living things. So the answer is (a)",
        "A belt buckle is made of metal. If a magnet is attracted to a metal then that magnet will stick to that metal. So the answer is (a)",
        "Claws are used by wolves to catch prey like deer. So the answer is (c)",
        "An electric car uses less gasoline than a regular car and thus causes less CO2 emissions. So the answer is (d)"
    ],
    'feedback': [
         {
            'sentence': "A plant requires sunlight for photosynthesis, which accumulates resources required for sprouting, blooming and wilting. So the answer is (d)",
            'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
        },
        {
            'sentence': "Mount Rushmore is a mountain. Mountains always seem closer when viewed from far away. So the answer is (a).",
            'feedback': "Factuality: Incorrect logic about mountains. 1/5\nRelevance: The statement is not entirely relevant to the question. 2/5\nInformativeness: Provides a misguided reasoning. 2/5\nTotal Score: 5/15"
        },
        {
            'sentence': "Digestion involves breaking down food. To digest a statement means to understand it. So the answer is (b).",
            'feedback': "Factuality: Misunderstanding of the term 'digest'. 2/5\nRelevance: It has some relevance due to the use of the term 'digest'. 2/5\nInformativeness: Provides a misleading explanation. 2/5\nTotal Score: 6/15"
        },
        {
            'sentence': "A tree is a living thing. Poison causes harm to living things. So the answer is (a)",
            'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
        },
        {
            'sentence': "Magnets can sometimes stick to surfaces. A wooden table is a surface. So the answer is (b).",
            'feedback': "Factuality: Incorrect understanding of magnetic materials. 2/5\nRelevance: The statement doesn't relate magnets to metals. 2/5\nInformativeness: The reasoning is misleading. 2/5\nTotal Score: 6/15"
        },
        {
            'sentence': "Wolves have fur which keeps them warm. Deer are afraid of warmth. So the answer is (a).",
            'feedback': "Factuality: Incorrect logic about deer's fear. 1/5\nRelevance: The statement is not directly relevant to deer's safety. 2/5\nInformativeness: Provides a misguided reasoning. 2/5\nTotal Score: 5/15"
        },
        {
            'sentence': "An electric car uses less gasoline than a regular car and thus causes less CO2 emissions. So the answer is (d)",
            'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
        }
    ],
    'refine':[
    [
        {
            'sentence': "The sun shines bright. When things are bright, puppies tend to learn new tricks. So the answer is (a).",
            'feedback': "Factuality: The assumption about puppies is incorrect. 1/5\nRelevance: The reasoning is not directly related to the sun's responsibility. 2/5\nInformativeness: Provides a misguided reasoning. 2/5\nTotal Score: 5/15"
        },
        {
            'sentence': "The sun causes children to go out and play. When children play, they grow. So the answer is (b).",
            'feedback': "Factuality: While sun encourages outdoor activities, it isn't the reason children grow. 2/5\nRelevance: It loosely relates the sun to children's growth. 3/5\nInformativeness: The sentence is misleading. 2/5\nTotal Score: 7/15"
        },
        {
            'sentence': "A plant requires sunlight for photosynthesis, which accumulates resources required for sprouting, blooming and wilting. So the answer is (d)",
            'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
        }
    ],
        [
        {
            'sentence': "Mount Rushmore is a mountain. Mountains always seem closer when viewed from far away. So the answer is (a).",
            'feedback': "Factuality: Incorrect logic about mountains. 1/5\nRelevance: The statement is not entirely relevant to the question. 2/5\nInformativeness: Provides a misguided reasoning. 2/5\nTotal Score: 5/15"
        },
        {
            'sentence': "Mountains, in general, can be less detailed from far away, but their size perception doesn't change. So the answer is (c).",
            'feedback': "Factuality: Inaccurate understanding of perspective. 2/5\nRelevance: It somewhat addresses the question about viewing the mountain from a distance. 3/5\nInformativeness: The sentence is somewhat informative. 3/5\nTotal Score: 8/15"
        },
        {
            'sentence': "When an object is far away, it takes up less of your field of view, and so seems smaller than in the photographs. So the answer is (d)",
            'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
        }
    ],
        [
        {
            'sentence': "When food is reduced in the stomach, people might turn to reading. Reader’s digest is a good read. So the answer is (d).",
            'feedback': "Factuality: Incorrect logic about food and reading. 1/5\nRelevance: The statement is not related to the digestion process. 1/5\nInformativeness: Misleading reasoning. 1/5\nTotal Score: 3/15"
        },
        {
            'sentence': "Digestion involves breaking down food. To digest a statement means to understand it. So the answer is (b).",
            'feedback': "Factuality: Misunderstanding of the term 'digest'. 2/5\nRelevance: It has some relevance due to the use of the term 'digest'. 2/5\nInformativeness: Provides a misleading explanation. 2/5\nTotal Score: 6/15"
        },
        {
            'sentence': "The stomach is part of the digestive system. The breaking down of food into nutrients occurs in the digestive system. So the answer is (c)",
            'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Directly relevant to the question. 5/5\nInformativeness: The sentence provides accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
        }
    ],
        
        [
        {
            'sentence': "Poison can corrode surfaces. A house is a surface. So the answer is (c).",
            'feedback': "Factuality: Incorrect understanding of poison's effects. 1/5\nRelevance: The statement is not directly relevant to the living entities. 2/5\nInformativeness: Provides a misguided reasoning. 2/5\nTotal Score: 5/15"
        },
        {
            'sentence': "Robots can malfunction if subjected to harmful substances. Poison is harmful. So the answer is (b).",
            'feedback': "Factuality: Incorrect relation between poison and robot malfunction. 2/5\nRelevance: Poison's harm to robots isn't a common understanding. 2/5\nInformativeness: The reasoning is misleading. 2/5\nTotal Score: 6/15"
        },
        {
            'sentence': "A tree is a living thing. Poison causes harm to living things. So the answer is (a)",
            'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
        }
    ],
    
    [
        {
            'sentence': "Magnets are attracted to light materials. A paper plate is light. So the answer is (d).",
            'feedback': "Factuality: Incorrect understanding of magnets. 1/5\nRelevance: The statement is not relevant to magnetism. 1/5\nInformativeness: Provides a misguided reasoning. 1/5\nTotal Score: 3/15"
        },
        {
            'sentence': "Magnets can sometimes stick to surfaces. A wooden table is a surface. So the answer is (b).",
            'feedback': "Factuality: Incorrect understanding of magnetic materials. 2/5\nRelevance: The statement doesn't relate magnets to metals. 2/5\nInformativeness: The reasoning is misleading. 2/5\nTotal Score: 6/15"
        },
        {
            'sentence': "A belt buckle is made of metal. If a magnet is attracted to a metal then that magnet will stick to that metal. So the answer is (a)",
            'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
        }
    ],
    
    [
        {
            'sentence': "Wolves have fur which keeps them warm. Deer are afraid of warmth. So the answer is (a).",
            'feedback': "Factuality: Incorrect logic about deer's fear. 1/5\nRelevance: The statement is not directly relevant to deer's safety. 2/5\nInformativeness: Provides a misguided reasoning. 2/5\nTotal Score: 5/15"
        },
        {
            'sentence': "Wolves are known to howl to communicate. Deer are scared of these howls. So the answer is (b).",
            'feedback': "Factuality: Over-simplified understanding of deer's fears. 3/5\nRelevance: It loosely relates wolf's howling to deer's fear. 3/5\nInformativeness: The sentence provides partial reasoning. 3/5\nTotal Score: 9/15"
        },
        {
            'sentence': "Claws are used by wolves to catch prey like deer. So the answer is (c)",
            'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
        }
    ],
    
    [
        {
            'sentence': "Electric cars use electricity. This electricity emits more CO2. So the answer is (a).",
            'feedback': "Factuality: Incorrect understanding of electric car emissions. 1/5\nRelevance: The statement is somewhat relevant but misleading. 3/5\nInformativeness: Provides a misguided reasoning. 2/5\nTotal Score: 6/15"
        },
        {
            'sentence': "Electric cars run on electric energy, so they cause electric emissions. So the answer is (c).",
            'feedback': "Factuality: Misunderstanding of the term 'electric emissions'. 2/5\nRelevance: The statement doesn't directly address CO2 emissions. 2/5\nInformativeness: The reasoning is misleading. 2/5\nTotal Score: 6/15"
        },
        {
            'sentence': "An electric car uses less gasoline than a regular car and thus causes less CO2 emissions. So the answer is (d)",
            'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."}
    ]
    ]
    },
    'qasc':{
    'init': [
       "Conserving resources has a positive impact on the environment. Use of resources affects the environment such as pollution. So the answer is (h).",
       "If a habitat can no longer support animals then those animals will move to another area. Cows are social animals. So the answer is (g).",
        "Microorganisms can cause infections. Infections usually require medical treatment. So the answer is (h).",
        "Healing requires rest. Lavender induces restful sleep. So the answer is (a).",
        "Freezing means changing from a liquid into a solid by reducing heat energy. Liquids freeze when they change to the solid state. So the answer is (f).",
        "Gametes then unite in fertilization and form a diploid zygote. Collectively, the sperm and the ova are also referred to as gametes. So the answer is (d).",
        "If an object is black then that object absorbs all visible light. Light grains are quartz, Black grains are coal. So the answer is (b)."
    ],
    'feedback': [
        {'sentence': "Burning fossil fuels generates energy. More energy means more work done. So the answer is (e).",
        'feedback': "Factuality: Incorrect understanding of fossil fuel impact. 1/5\nRelevance: The statement is related but misleading. 2/5\nInformativeness: Provides incorrect reasoning. 1/5\nTotal Score: 4/15"},
        {'sentence': "If a habitat can no longer support animals then those animals will move to another area. Cows are social animals. So the answer is (g).",
        'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."},
        {'sentence': "Microorganisms can cause infections. Infections usually require medical treatment. So the answer is (h).",
        'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."},
        {'sentence': "Lavender can induce warmth due to its calming effects. So the answer is (h).",
        'feedback': "Factuality: Partially correct but not the best answer. 3/5\nRelevance: The statement is relevant but not entirely correct. 3/5\nInformativeness: Information is not sufficient for the answer. 3/5\nTotal Score: 9/15"},
        {'sentence': "When frozen, a liquid turns into a gas because it loses its liquid properties. So the answer is (c).",
        'feedback': "Factuality: Incorrect information. 1/5\nRelevance: Not relevant to the question. 1/5\nInformativeness: Does not provide correct information. 1/5\nTotal Score: 3/15"},
        {'sentence': "Most plants unite to form a diploid zygote for reproduction. So the answer is (b).",
        'feedback': "Factuality: Incorrect as plants don't unite to form zygotes. 1/5\nRelevance: Not relevant to the question. 1/5\nInformativeness: Provides incorrect reasoning. 1/5\nTotal Score: 3/15"},
        {'sentence': "Skin absorbs all visible light because it covers the entire body. So the answer is (e).",
        'feedback': "Factuality: Incorrect understanding of light absorption. 2/5\nRelevance: Is relevant to the question. 3/5\nInformativeness: Does not provide correct information. 1/5\nTotal Score: 6/15"}
    ],
    'refine':[
    [
            {
                'sentence': "Burning fossil fuels generates energy. More energy means more work done. So the answer is (e).",
                'feedback': "Factuality: Incorrect understanding of fossil fuel impact. 1/5\nRelevance: The statement is related but misleading. 2/5\nInformativeness: Provides incorrect reasoning. 1/5\nTotal Score: 4/15"
            },
            {
                'sentence': "Using transportation technology reduces the time of travel, thus reducing pollution. So the answer is (b).",
                'feedback': "Factuality: Partially correct, but lacks understanding. 3/5\nRelevance: The statement is relevant but not entirely correct. 3/5\nInformativeness: Provides partial information. 3/5\nTotal Score: 9/15"
            },
            {
                'sentence': "Conserving resources has a positive impact on the environment. Use of resources affects the environment such as pollution. So the answer is (h).",
                'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
            }
        ],
        [
            {
                'sentence': "Birds can fly to other areas easily. So the answer is (b).",
                'feedback': "Factuality: It is a true characteristic of birds, but not answering the question. 2/5\nRelevance: Somewhat related to the question but not entirely correct. 2/5\nInformativeness: Provides minimal information. 2/5\nTotal Score: 6/15"
            },
            {
                'sentence': "Humans can move to other areas when their habitat is no longer supportive. So the answer is (d).",
                'feedback': "Factuality: True for humans but not a general answer. 3/5\nRelevance: Relevant to the question but not the best answer. 3/5\nInformativeness: Provides some information. 3/5\nTotal Score: 9/15"
            },
            {
                'sentence': "If a habitat can no longer support animals then those animals will move to another area. Cows are social animals. So the answer is (g).",
                'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
            }
        ],
    [
            {
                'sentence': "Contact with latex can cause skin issues requiring medical attention. So the answer is (a).",
                'feedback': "Factuality: Incorrect as the question excepts allergies. 1/5\nRelevance: The statement is related but not correct. 2/5\nInformativeness: Provides incorrect reasoning. 1/5\nTotal Score: 4/15"
            },
            {
                'sentence': "A tree falling can cause injuries requiring medical attention. So the answer is (b).",
                'feedback': "Factuality: Factually correct but not the best answer. 3/5\nRelevance: Relevant to the question but not the best answer. 3/5\nInformativeness: Provides partial information. 3/5\nTotal Score: 9/15"
            },
            {
                'sentence': "Microorganisms can cause infections. Infections usually require medical treatment. So the answer is (h).",
                'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
            }
        ],

    [
            {
                'sentence': "Lavender can induce mutations because of its genetic properties. So the answer is (d).",
                'feedback': "Factuality: Incorrect understanding of lavender properties. 1/5\nRelevance: The statement is related but not correct. 2/5\nInformativeness: Provides incorrect reasoning. 1/5\nTotal Score: 4/15"
            },
            {
                'sentence': "Lavender can induce warmth due to its calming effects. So the answer is (h).",
                'feedback': "Factuality: Partially correct but not the best answer. 3/5\nRelevance: The statement is relevant but not entirely correct. 3/5\nInformativeness: Information is not sufficient for the answer. 3/5\nTotal Score: 9/15"
            },
            {
                'sentence': "Healing requires rest. Lavender induces restful sleep. So the answer is (a).",
                'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
            }
        ],
    [
            {
                'sentence': "When frozen, a liquid turns into a gas because it loses its liquid properties. So the answer is (c).",
                'feedback': "Factuality: Incorrect information. 1/5\nRelevance: Not relevant to the question. 1/5\nInformativeness: Does not provide correct information. 1/5\nTotal Score: 3/15"
            },
            {
                'sentence': "When frozen, a liquid is in a cold state because the temperature is low. So the answer is (h).",
                'feedback': "Factuality: Semi-correct but phrased wrongly. 3/5\nRelevance: Somewhat related to the question but not entirely correct. 3/5\nInformativeness: Provides minimal information. 2/5\nTotal Score: 8/15"
            },
            {
                'sentence': "Freezing means changing from a liquid into a solid by reducing heat energy. Liquids freeze when they change to the solid state. So the answer is (f).",
                'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
            }
        ],
    [
            {
                'sentence': "Most plants unite to form a diploid zygote for reproduction. So the answer is (b).",
                'feedback': "Factuality: Incorrect as plants don't unite to form zygotes. 1/5\nRelevance: Not relevant to the question. 1/5\nInformativeness: Provides incorrect reasoning. 1/5\nTotal Score: 3/15"
            },
            {
                'sentence': "Predator and prey uniting can form a diploid zygote in some cases. So the answer is (f).",
                'feedback': "Factuality: Somewhat correct 3/5\nRelevance: Is relevant to the question 4/5\nInformativeness: There is some partial wrongness in the answer 2/5\nTotal Score: 9/15"
            },
            {
                'sentence': "Gametes then unite in fertilization and form a diploid zygote. Collectively, the sperm and the ova are also referred to as gametes. So the answer is (d).",
                'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
            }
        ],
    [
            {
                'sentence': "Glass absorbs all visible light making it difficult to see through. So the answer is (g).",
                'feedback': "Factuality: Incorrect as glass does not absorb all visible light. 1/5\nRelevance: somewhat relevant to the question. 3/5\nInformativeness: Provides incorrect reasoning. 1/5\nTotal Score: 5/15"
            },
            {
                'sentence': "Skin absorbs all visible light because it covers the entire body. So the answer is (e).",
                'feedback': "Factuality: Incorrect understanding of light absorption. 2/5\nRelevance: Is relevant to the question. 3/5\nInformativeness: Does not provide correct information. 1/5\nTotal Score: 6/15"
            },
            {
                'sentence': "If an object is black then that object absorbs all visible light. Light grains are quartz, Black grains are coal. So the answer is (b).",
                'feedback': "Factuality: Sentence is factual. 5/5\nRelevance: Sentence is directly relevant to the question. 5/5\nInformativeness: The sentence provides accurate reasoning. 5/5\nTotal Score: 15/15. Stop refining the answer."
            }
        ]
    ]
    }
}

templates = {'init': 'A: {answer}',
             'feedback': 'A: {answer}\nScores:\n{feedback}',
             'refine':'A: {answer}\nScores:\n{feedback}'}

target_templates  = {'init': 'Q: {question}\nAnswer choices:\n{choices}\nA: ',
                     'feedback':'Q: {question}\nAnswer choices:\n{choices}\n A: {answer}\nScores:',
                     'refine':'Q: {question}\nAnswer choices:\n{choices}\nA: {answer}\nScores:\n{scores}\n\nOkay, improve the sentence using the feedback:\n\n'}

instructions = {
    'init':"Given a question with answer choices, generate a reasoning explanation which supports the selected answer. Desired traits for the reasoning are explanation are 1) Factuality - The reasoning should be factual and should not contain any false information. 2) Relevance - The reasoning should be relevant to both the question and answer. 3) Informativeness - The reasoning should provide sufficient information to support the answer.",
    'feedback':"We want to iteratively improve the provided responses. To help improve, scores for each response on desired traits are provided: 1) Factuality, 2) Relevance, 3) Informativeness. Please rate each trait from 1 to 5 and decide if the answer requires further refinement. If not, append 'stop refining the answer' to the end of the feedback.",
    'refine':"We want to iteratively improve the provided responses. To help improve, scores for each response on desired traits are provided: 1) Factuality, 2) Relevance, 3) Informativeness. Please rate each trait from 1 to 5 and decide if the answer requires further refinement. If not, append 'stop refining the answer' to the end of the feedback."
}
refine_instr = "\n\nOkay, improve the sentence using the feedback:\n\n"
inter_example_sep="\n\n"

def format_refine_prompt(prompt,explanation = None,stage='init',num_shot = 10,dataset = 'obqa'):
    """
    prompt is a list containing question,choices for init, additional answer for feedback and and additional feedback for refine
    choices is the target choices
    fs_prompt are the few-shot question,choices and answers
    explanation is only used during eval, when the perturbated explanation is given
    stage is either [init, feedback, refine]
    shot = how many shots
    dataset = which dataset to select refine prompt
    
    Return completed prompt for the stage.
    Does not include the instruction for the 3 stages. add in the separate fn.
    """
    starting_prompt = []
    refine_prompt = refine_template[dataset]
    fs_prompt = cot_template[dataset]
    
    random_ids = np.arange(len(fs_prompt['Q']))
    # if num_shot < len(fs_prompt['Q']):
    #     np.random.shuffle(random_ids)
    max_shot_ids = random_ids[:num_shot].tolist()

    for i in max_shot_ids:
        q = fs_prompt['Q'][i]
        c = fs_prompt['choices'][i]
        curr_p = []
        curr_p.append(f"Q: {q}")
        curr_p.append('Answer choices:\n'+'\n'.join(c))
        if stage == 'init':
            curr_p.append(templates[stage].format(answer = refine_prompt['init'][i]))
        elif stage == 'feedback':
            curr_p.append(templates[stage].format(answer = refine_prompt['feedback'][i]['sentence'],feedback = refine_prompt['feedback'][i]['feedback']))
        elif stage == 'refine':
            sub_p = []
            for refine in refine_prompt['refine'][i]:
                sub_p.append(templates['feedback'].format(answer = refine['sentence'],feedback = refine['feedback']))
            curr_p.append(refine_instr.join(sub_p))
        starting_prompt.append('\n'.join(curr_p))
    starting_prompt = inter_example_sep.join(starting_prompt)
    
    tar_question = prompt['question']
    tar_choices = prompt['choices']
    tar_answer = prompt.get('answer',None)
    tar_feedback = prompt.get('feedback',None)
    
    num_to_alpha =  {str(i):chr(ord('a') + i) for i in range(len(tar_choices))}
    starting_s = []
    for i,choice in enumerate(tar_choices):
        starting_s.append(f"({num_to_alpha[str(i)]}) {choice}")
    target_choice = '\n'.join(starting_s)
    
    out_prompt = instructions[stage] + '\n\n' + starting_prompt + inter_example_sep 
    if stage == 'init':
        out_prompt += target_templates['init'].format(question = tar_question,choices = target_choice)
    elif stage == 'feedback':
        out_prompt += target_templates['feedback'].format(question = tar_question,choices = target_choice,answer = tar_answer)
    else:
        out_prompt += target_templates['refine'].format(question = tar_question,choices = target_choice,answer = tar_answer,scores = tar_feedback)
    return out_prompt
    

def process_answer(texts,num_choices=4): 
    """
    Given a list of texts where the generated output in text form,
    return the answer
    Can be use for init and refine stage.
    return list of (answer in str and ans index) -> index used for final answer, str use for iteration
    """
    
    ans_pattern = r'\(([a-zA-Z])\)'
    alpha_to_num = {chr(ord('a') + i):i for i in range(num_choices)}
    out_str,out_id,out_expl = [],[],[]
    for text in texts:
        ans_sen = ''
        text_split = text.lstrip().split('\n')
        first_sen = text_split[0].strip()
        if first_sen.startswith('A:'):
            ans_sen = first_sen.split('A:')[1].strip()
        else:
            ans_sen = first_sen
        ans_match = re.search(ans_pattern,ans_sen)
        if ans_match:
            ans = ans_match.group(1)
            ans_num = alpha_to_num[ans]
            out_str.append(ans_sen)
            out_id.append(ans_num)
        else:
            out_str.append(ans_sen)
            out_id.append(-1)
        if 'So the answer' in ans_sen:
            expl = ans_sen.split('So the answer')[0].strip()
            out_expl.append(expl)
        else:
            out_expl.append(ans_sen)
    return out_str,out_id,out_expl

def process_feedback(texts): 
    """
    Given the generated output or input prompt in text form,
    process the answer and extract the feedback
    return the feedback and stop condition -> tuple
    """
    out_fb,out_stop = [],[]
    end_text = 'total score'
    end_id = 0
    for text in texts:
        stop = False
        for text_id,t in enumerate(text.split('\n')):
            if end_text in t or '/15' in t:
                end_id = text_id
                if 'stop refining' in t.lower():
                    stop = True
                break
        if end_id != 0:
            out_fb.append('\n'.join(text.split('\n')[:end_id+1]))
            out_stop.append(stop)
        else:
            out_fb.append(text)
            out_stop.append(False)
    return out_fb,out_stop

            
            
            
    