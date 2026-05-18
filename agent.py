"""
ASSIGNMENT 2 – DEL 1: ReAct Agent
==================================
ML1-kursen, Teknikhögskolan

Vad detta program gör:
    Detta är en ReAct-agent (Reasoning + Acting) byggd i ren Python.
    Agenten tar emot en uppgift i naturligt språk från användaren,
    låter en AI-modell (Claude) resonera om vad som behöver göras,
    kör bash-kommandon på datorn, och matar tillbaka resultaten
    till modellen tills uppgiften är löst.

ReAct-mönstret (cykeln som upprepas):
    THOUGHT  → modellen tänker högt om vad den ska göra
    ACTION   → modellen väljer att köra ett bash-kommando
    COMMAND  → det specifika kommandot (t.ex. "ls -la")
    OBSERVATION → resultatet av kommandot, matas tillbaka till modellen
    ANSWER   → modellens slutsvar när uppgiften är löst

Viktigt för Del 1:
    - Vi använder INTE inbyggd function-calling (tools-parametern i API:t)
    - Vi använder INGA ramverk (LangChain, LangGraph, etc.)
    - Istället instruerar vi modellen via en system-prompt att svara i
      ett specifikt textformat, och parsar sedan svaret med egen kod (regex)
    - Detta är "hemmagjord function-calling" – poängen är att vi ska
      förstå att function-calling egentligen bara är prompt engineering + parsing
"""

# ── Imports ──────────────────────────────────────────────────────────────────
# os: läser miljövariabler (vår API-nyckel)
# re: regex – söker efter mönster i text (t.ex. "COMMAND: ls -la")
# subprocess: kör bash-kommandon från Python och fångar resultatet
import os
import re
import subprocess

# anthropic: Anthropics officiella SDK för att prata med Claude API
# dotenv: läser in .env-filen så att API-nyckeln finns som miljövariabel
import anthropic
from dotenv import load_dotenv

# Ladda ANTHROPIC_API_KEY från .env-filen i projektmappen.
# Utan detta måste man exportera nyckeln manuellt i terminalen varje gång.
load_dotenv()

# ── Modellval ────────────────────────────────────────────────────────────────

# Vi använder Claude Sonnet 4.6 för kostnadseffektivitet.
# Agenten gör MÅNGA API-anrop i sin loop (ett per steg),
# så Opus 4.7 (som är smartare men dyrare) skulle kosta onödigt mycket.
# Sonnet är mer än kapabel nog för att resonera och föreslå bash-kommandon.
MODEL = "claude-sonnet-4-6"

# ── System-prompt ─────────────────────────────────────────────────────────────

# System-prompten är KÄRNAN i hela agenten.
# Den skickas som "system"-parametern i API-anropet (inte som ett vanligt meddelande).
# Den instruerar modellen att ALLTID svara i ett specifikt format som vi kan parsa.
# Utan denna prompt skulle Claude svara i vanlig prosa och vi kunde inte
# plocka ut kommandon automatiskt.
#
# Formatet vi definierar:
#   - THOUGHT/ACTION/COMMAND → modellen vill köra ett kommando
#   - ANSWER → modellen är klar och ger sitt slutsvar
#
# Reglerna i prompten är resultatet av buggfixar under utvecklingen:
#   - "CRITICAL: Never include both COMMAND and ANSWER" lades till efter att
#     modellen blandade båda i samma svar, vilket fick agenten att hoppa
#     över kommandoexekveringen (se rapport sektion 7.1)
#   - "Do NOT assume the output" lades till för att förhindra att modellen
#     hittar på vad ett kommando returnerar istället för att vänta på det riktiga resultatet

SYSTEM_PROMPT = """You are a helpful assistant that can run bash commands to answer questions.

When you need to run a command, respond in EXACTLY this format (nothing before or after):

THOUGHT: <your reasoning about what to do next>
ACTION: bash
COMMAND: <the bash command to run>

When you have a final answer and no more commands are needed, respond in EXACTLY this format:

ANSWER: <your complete answer to the user>

Rules:
- CRITICAL: Never include both COMMAND and ANSWER in the same response.
- Always wait for the OBSERVATION before giving an ANSWER.
- Only use THOUGHT/ACTION/COMMAND or ANSWER — never mix them in the same response.
- Do NOT assume the output of a command. Always wait for the actual OBSERVATION.
- Keep commands simple and safe.
- If a command fails, try a different approach.
"""

# ── Parsningsfunktioner ──────────────────────────────────────────────────────
# Dessa funktioner är vår "hemmagjorda function-calling".
# De letar efter specifika mönster i modellens textsvar
# och plockar ut den information vi behöver (kommando eller slutsvar).
# I Del 2 kommer vi byta detta mot structured output (JSON),
# men i Del 1 är kravet att vi gör det själva med stränghantering.


def parse_command(text: str) -> str | None:
    """
    Plocka ut bash-kommandot från modellens svar.

    Letar efter en rad som börjar med "COMMAND:" och returnerar
    allt som kommer efter. T.ex. om modellen svarar:
        THOUGHT: Jag behöver lista filer
        ACTION: bash
        COMMAND: ls -la
    ...så returnerar denna funktion strängen "ls -la".

    Regex-mönstret förklarat:
        ^          = radens början (tack vare re.MULTILINE)
        COMMAND:   = den exakta texten vi letar efter
        \\s*        = noll eller fler mellanslag efter kolon
        (.+)       = fånga resten av raden (detta är kommandot)
        $          = radens slut

    Returnerar None om inget kommando hittades.
    """
    match = re.search(r"^COMMAND:\s*(.+)$", text, re.MULTILINE)
    if match:
        # group(1) ger oss det som fångades av (.+), alltså själva kommandot
        return match.group(1).strip()
    return None


def is_final_answer(text: str) -> bool:
    """
    Kolla om modellens svar innehåller ett slutsvar.

    Letar efter en rad som börjar med "ANSWER:".
    Returnerar True om den hittar det, annars False.
    """
    return bool(re.search(r"^ANSWER:", text, re.MULTILINE))


def extract_answer(text: str) -> str:
    """
    Plocka ut själva svarstexten som kommer efter "ANSWER:".

    re.DOTALL gör att . matchar även radbrytningar,
    så hela svaret fångas även om det sträcker sig över flera rader.

    Om inget ANSWER-mönster hittas returneras hela texten som fallback.
    """
    match = re.search(r"^ANSWER:\s*(.+)", text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text  # fallback: returnera allt


# ── Kommandoexekvering med säkerhet ──────────────────────────────────────────
# Säkerhet är kritiskt! En agent som kör bash-kommandon kan potentiellt:
#   - Radera filer (rm -rf /)
#   - Installera skadlig kod (curl ... | bash)
#   - Eskalera privilegier (sudo)
#   - Skicka iväg känslig data
#   - Orsaka stora kostnader via oändliga loopar
#
# I Del 1 skyddar vi oss med:
#   1. Manuell y/n-bekräftelse innan varje kommando
#   2. Timeout på 30 sekunder
#   3. max_tokens-begränsning i API-anropet
#   4. stop_sequences som hindrar modellen från att fabricera resultat
#
# I Del 2 kommer vi lägga till blocklista mot destruktiva kommandon.


def ask_user_confirmation(command: str) -> bool:
    """
    Visa kommandot för användaren och fråga om det ska köras.

    Detta är vår primära säkerhetsmekanism i Del 1.
    Användaren ser exakt vad agenten vill göra och kan neka.
    Returnerar True om användaren svarar 'y' eller 'yes'.
    """
    print(f"\n  Command to run: {command}")
    answer = input("  Run this command? [y/n]: ").strip().lower()
    return answer in ("y", "yes")


def run_command(command: str) -> str:
    """
    Kör ett bash-kommando och returnera dess output.

    Parametrar till subprocess.run():
        shell=True       → kör genom /bin/sh så att pipes (|),
                           redirects (>), och andra shell-funktioner fungerar
        capture_output   → fånga både stdout (normal output)
                           och stderr (felmeddelanden)
        text=True        → konvertera bytes till strängar automatiskt
        timeout=30       → avbryt efter 30 sek om kommandot hänger sig

    Vi slår ihop stdout och stderr till en sträng som returneras.
    Om kommandot inte ger någon output returneras "(no output)" –
    detta är viktigt eftersom modellen behöver något att reagera på.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Slå ihop stdout och stderr så modellen ser allt (inklusive felmeddelanden)
        output = result.stdout + result.stderr
        return output.strip() if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 30 seconds"
    except Exception as e:
        return f"ERROR: {e}"


# ── Agentloopen (ReAct-loopen) ───────────────────────────────────────────────
# Detta är HJÄRTAT i hela programmet.
# Flödet:
#   1. Skicka konversationshistoriken till Claude API
#   2. Parsa svaret – finns det ett COMMAND? Kör det.
#   3. Finns det ett ANSWER? Vi är klara.
#   4. Lägg till modellens svar + kommandots resultat i historiken
#   5. Gå tillbaka till steg 1
#
# Konversationshistoriken (messages-listan) VÄXER för varje steg.
# Det är så modellen "minns" vad som hänt – vi skickar ALLT varje gång.


def run_agent(user_task: str) -> None:
    """
    Kör ReAct-loopen för en given uppgift.

    Tar emot en uppgift som sträng (t.ex. "List all Python files"),
    kör loopen tills modellen ger ett ANSWER, och skriver ut resultatet.
    Returnerar ingenting (None) – all output skrivs direkt till terminalen.
    """
    # Skapa en klient som kan prata med Claude API.
    # API-nyckeln hämtas från miljövariabeln som load_dotenv() laddade in.
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Starta konversationshistoriken med användarens uppgift.
    # Denna lista kommer växa för varje steg i loopen:
    #   Varv 1: [user: uppgift]
    #   Varv 2: [user: uppgift, assistant: THOUGHT+COMMAND, user: OBSERVATION]
    #   Varv 3: [user: uppgift, assistant: THOUGHT+COMMAND, user: OBSERVATION, assistant: ANSWER]
    messages = [
        {"role": "user", "content": user_task},
    ]

    print("\n" + "─" * 60)
    print(f"Task: {user_task}")
    print("─" * 60)

    # Räknare för att visa vilket steg vi är på
    step = 1

    # ── ReAct-loopen börjar här ──────────────────────────────────────────
    # Denna loop körs tills den bryts inifrån (via break).
    # Varje varv = ett API-anrop + eventuell kommandoexekvering.
    while True:
        print(f"\n[Step {step}] Asking model...")

        # ── STEG 1: Anropa Claude API ────────────────────────────────────
        # Skicka hela konversationshistoriken till modellen.
        # Notera:
        #   - system=SYSTEM_PROMPT → instruerar modellen om formatet
        #   - messages=messages → hela historiken (växer varje varv)
        #   - INGEN tools-parameter → vi använder INTE inbyggd function-calling
        #   - stop_sequences=["OBSERVATION"] → VIKTIGT: stoppar modellen om
        #     den försöker generera en OBSERVATION själv. Vi vill att
        #     observationen kommer från det riktiga kommandoresultatet,
        #     inte från modellens fantasi.
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
            stop_sequences=["OBSERVATION"],
        )

        # Plocka ut modellens textsvar.
        # response.content är en lista med block – vi tar det första.
        model_text = response.content[0].text.strip()
        print(f"\n{model_text}")

        # ── STEG 2: Kolla efter COMMAND först ─────────────────────────────
        # Vi kollar efter COMMAND FÖRE ANSWER. Varför?
        # Bugg vi hittade: modellen blandade ibland COMMAND och ANSWER
        # i samma svar. Om vi kollade ANSWER först hoppade vi över
        # kommandoexekveringen. Genom att prioritera COMMAND säkerställer
        # vi att kommandon alltid körs innan vi avslutar.
        command = parse_command(model_text)

        if command is not None:
            # Modellen vill köra ett kommando – fråga användaren först (säkerhet!)
            if ask_user_confirmation(command):
                observation = run_command(command)
                print(f"\n  OBSERVATION: {observation}")
            else:
                observation = "User declined to run the command."
                print(f"\n  OBSERVATION: {observation}")

            # Lägg till utbytet i konversationshistoriken:
            # - Modellens svar (THOUGHT+COMMAND) som "assistant"
            # - Kommandots resultat som "user" med prefix "OBSERVATION:"
            # Nästa varv skickas allt detta tillbaka till API:t
            # så modellen ser vad kommandot returnerade.
            messages.append({"role": "assistant", "content": model_text})
            messages.append(
                {"role": "user", "content": f"OBSERVATION: {observation}"})
            step += 1
            # continue = gå tillbaka till toppen av while-loopen
            # (nästa varv i ReAct-cykeln)
            continue

        # ── STEG 3: Kolla efter slutsvar ─────────────────────────────────
        # Bara om det INTE fanns något kommando kollar vi efter ANSWER.
        # Om modellen svarade med "ANSWER:" är uppgiften löst – bryt loopen.
        if is_final_answer(model_text):
            print("\n" + "─" * 60)
            print("FINAL ANSWER:")
            print(extract_answer(model_text))
            print("─" * 60)
            break

        # ── STEG 4: Fallback ─────────────────────────────────────────────
        # Om modellen varken svarade med COMMAND eller ANSWER
        # (t.ex. svarade med en snygg tabell i fritext utan prefix)
        # behandla hela svaret som ett slutsvar istället för att krascha.
        # Detta är defensiv programmering – LLM:er följer inte alltid
        # instruktioner exakt.
        print("\n" + "─" * 60)
        print("FINAL ANSWER:")
        print(model_text)
        print("─" * 60)
        break


# ── Programmets startpunkt ────────────────────────────────────────────────────


def main() -> None:
    """
    Programmets huvudfunktion. Visar en rubrik, frågar efter uppgifter
    i en loop, och skickar varje uppgift till run_agent().
    Returnerar ingenting (-> None).
    """
    print("ReAct Agent (type 'quit' to exit)")
    print("=" * 60)

    # Oändlig loop som frågar efter uppgifter tills användaren avslutar
    while True:
        try:
            task = input("\nYour task: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Fånga Ctrl+C och Ctrl+D för att avsluta snyggt
            # istället för att programmet kraschar med en traceback
            print("\nGoodbye!")
            break

        # Tre sätt att avsluta: quit, exit, eller q
        if task.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Om användaren trycker Enter utan att skriva något – fråga igen
        if not task:
            continue

        # Skicka uppgiften till agentloopen.
        # När run_agent() är klar (ANSWER hittat) kommer vi tillbaka hit
        # och frågar "Your task:" igen.
        run_agent(task)


# Standardmönster i Python:
# "Om denna fil körs direkt (python3 agent.py), starta main().
#  Om filen importeras från en annan fil, gör ingenting."
if __name__ == "__main__":
    main()
