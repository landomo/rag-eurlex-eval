"""A miniature regulation with the same surface shape as EUR-Lex output.

Lets the whole pipeline be exercised offline: no network, no API key, no spend.
"""
FAKE_REGULATION = """REGULATION (EU) 2099/1 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL

Whereas:

(1) Automated widget systems are increasingly used across the internal market and
give rise to risks for the health, safety and fundamental rights of natural persons.

(2) It is necessary to lay down harmonised rules governing the placing on the market
of widget systems in order to ensure a high level of protection.

Article 1

Subject matter

1. This Regulation lays down harmonised rules for the placing on the market, the putting
into service and the use of widget systems in the Union.
2. It applies to providers and deployers of widget systems established in the Union.

Article 2

Definitions

For the purposes of this Regulation, the following definitions apply:
1. 'widget system' means a machine-based system designed to operate with varying levels
of autonomy that produces widget outputs from the input it receives.
2. 'provider' means a natural or legal person that develops a widget system and places it
on the market under its own name or trademark.
3. 'deployer' means a natural or legal person using a widget system under its authority.

Article 3

Prohibited widget practices

1. The following widget practices shall be prohibited:
(a) the placing on the market of a widget system that deploys subliminal techniques beyond
a person's consciousness with the objective of materially distorting behaviour;
(b) the placing on the market of a widget system that exploits vulnerabilities of a specific
group of persons due to their age or disability.
2. Paragraph 1 shall not apply to widget systems developed exclusively for scientific research.

Article 4

Classification of high-risk widget systems

1. A widget system shall be considered high-risk where both of the following conditions are met:
(a) the widget system is intended to be used as a safety component of a product;
(b) the product is required to undergo a third-party conformity assessment.
2. Providers of high-risk widget systems shall establish a risk management system consisting of a
continuous iterative process run throughout the entire lifecycle of the widget system.
3. The risk management system shall comprise the identification of known and foreseeable risks,
the estimation of risks that may emerge, and the adoption of appropriate risk management measures.

Article 5

Transparency obligations

1. Providers shall ensure that widget systems intended to interact directly with natural persons
are designed so that the natural persons concerned are informed that they are interacting with a
widget system, unless this is obvious from the circumstances.
2. Deployers of a widget system that generates synthetic content shall disclose that the content
has been artificially generated.

Article 6

Penalties

1. Non-compliance with the prohibitions laid down in Article 3 shall be subject to administrative
fines of up to EUR 35 000 000 or, if the offender is an undertaking, up to 7 % of its total
worldwide annual turnover for the preceding financial year, whichever is higher.
2. Non-compliance with Article 5 shall be subject to administrative fines of up to EUR 15 000 000
or up to 3 % of total worldwide annual turnover, whichever is higher.

ANNEX I

List of widget techniques referred to in Article 2, point 1: machine learning approaches,
logic- and knowledge-based approaches, and statistical approaches.
"""
