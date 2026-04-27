import random
import csv
from dataclasses import dataclass


# ============================================================
# Problema: SATURN + URANUS + NEPTUNE + PLUTO = PLANETS
# ============================================================

LETTERS = ["S", "A", "T", "U", "R", "N", "E", "P", "L", "O"]

ADDENDS = ["SATURN", "URANUS", "NEPTUNE", "PLUTO"]
RESULT = "PLANETS"

LEADING_LETTERS = ["S", "U", "N", "P"]


@dataclass
class Config:
    population_size: int = 3000
    max_generations: int = 5000
    crossover_probability: float = 0.90
    mutation_probability: float = 0.35
    tournament_size: int = 5
    elitism: int = 10
    number_of_runs: int = 3


# ============================================================
# 1. Decodificación del cromosoma
# ============================================================
# Cromosoma:
# Es una permutación de los dígitos 0..9.
#
# Usamos las 10 posiciones para asignar:
#
# [S, A, T, U, R, N, E, P, L, O]
#
# Ejemplo solución:
#
# [1, 2, 7, 5, 0, 3, 9, 4, 6, 8]
#
# Significa:
#
# S=1, A=2, T=7, U=5, R=0, N=3, E=9, P=4, L=6, O=8
#
# En este problema no hay posiciones sobrantes porque se usan
# exactamente 10 letras distintas.


def decode(individual):
    return {letter: digit for letter, digit in zip(LETTERS, individual)}


def build_number(mapping, word):
    value = 0

    for ch in word:
        value = value * 10 + mapping[ch]

    return value


# ============================================================
# 2. Función de aptitud
# ============================================================
# El objetivo es:
#
# SATURN + URANUS + NEPTUNE + PLUTO = PLANETS
#
# Minimizamos:
#
# |SATURN + URANUS + NEPTUNE + PLUTO - PLANETS|
#
# Además:
# - penalizamos si alguna palabra empieza con 0.
# - agregamos una penalización por errores columna a columna.
#
# La penalización columna a columna ayuda al AG a tener más información
# sobre qué tan cerca está una solución, no solo por el resultado final.


def column_error(mapping):
    carry = 0
    error = 0

    max_len = len(RESULT)

    for pos in range(max_len):
        column_sum = carry

        for word in ADDENDS:
            if pos < len(word):
                column_sum += mapping[word[-1 - pos]]

        expected_digit = mapping[RESULT[-1 - pos]]

        error += abs((column_sum % 10) - expected_digit)

        carry = column_sum // 10

    error += carry * 10

    return error


def evaluate(individual):
    mapping = decode(individual)

    addends_value = sum(build_number(mapping, word) for word in ADDENDS)
    result_value = build_number(mapping, RESULT)

    numeric_error = abs(addends_value - result_value)

    penalty = 0

    for letter in LEADING_LETTERS:
        if mapping[letter] == 0:
            penalty += 10_000_000

    column_penalty = column_error(mapping) * 100_000

    return numeric_error + column_penalty + penalty


# ============================================================
# 3. Generación de población inicial
# ============================================================

def create_individual():
    digits = list(range(10))
    random.shuffle(digits)
    return digits


def create_initial_population(size):
    return [create_individual() for _ in range(size)]


# ============================================================
# 4. Selección
# ============================================================
# Usamos selección por torneo.
# Como estamos minimizando, gana el individuo con menor fitness.


def tournament_selection(population, fitness_values, tournament_size):
    selected = []
    population_size = len(population)

    for _ in range(population_size):
        candidate_indexes = random.sample(range(population_size), tournament_size)

        winner_index = min(
            candidate_indexes,
            key=lambda i: fitness_values[i]
        )

        selected.append(population[winner_index][:])

    return selected


# ============================================================
# 5. Cruzamiento
# ============================================================
# Usamos Ordered Crossover porque el cromosoma es una permutación.
# Esto evita generar cromosomas con dígitos repetidos.


def ordered_crossover(parent1, parent2):
    size = len(parent1)

    start, end = sorted(random.sample(range(size), 2))

    child1 = [None] * size
    child2 = [None] * size

    child1[start:end + 1] = parent1[start:end + 1]
    child2[start:end + 1] = parent2[start:end + 1]

    def fill_child(child, donor):
        current_pos = (end + 1) % size
        donor_order = donor[end + 1:] + donor[:end + 1]

        for gene in donor_order:
            if gene not in child:
                child[current_pos] = gene
                current_pos = (current_pos + 1) % size

        return child

    child1 = fill_child(child1, parent2)
    child2 = fill_child(child2, parent1)

    return child1, child2


# ============================================================
# 6. Mutación
# ============================================================
# Mutación por intercambio de posiciones.
#
# Como el cromosoma es una permutación, no cambiamos un dígito por otro
# cualquiera porque podríamos repetir valores.
#
# En cambio, intercambiamos dos posiciones.


def swap_mutation(individual):
    i, j = random.sample(range(len(individual)), 2)

    individual[i], individual[j] = individual[j], individual[i]

    return individual


# ============================================================
# 7. Ejecución de una corrida
# ============================================================

def run_genetic_algorithm(config, seed):
    random.seed(seed)

    population = create_initial_population(config.population_size)

    best_individual = None
    best_fitness = float("inf")
    best_generation = 0

    log = []

    for generation in range(config.max_generations + 1):
        # Evaluación de la población
        fitness_values = [evaluate(individual) for individual in population]

        generation_best_index = min(
            range(len(population)),
            key=lambda i: fitness_values[i]
        )

        generation_best_individual = population[generation_best_index][:]
        generation_best_fitness = fitness_values[generation_best_index]

        if generation_best_fitness < best_fitness:
            best_fitness = generation_best_fitness
            best_individual = generation_best_individual[:]
            best_generation = generation

        avg_fitness = sum(fitness_values) / len(fitness_values)
        worst_fitness = max(fitness_values)

        log.append({
            "run_seed": seed,
            "generation": generation,
            "best_generation_fitness": generation_best_fitness,
            "avg_generation_fitness": avg_fitness,
            "worst_generation_fitness": worst_fitness,
            "best_global_fitness": best_fitness,
            "best_global_individual": best_individual[:]
        })

        # Criterio de paro
        if best_fitness == 0:
            break

        # Elitismo: conservamos los mejores individuos
        elite_indexes = sorted(
            range(len(population)),
            key=lambda i: fitness_values[i]
        )[:config.elitism]

        elites = [population[i][:] for i in elite_indexes]

        # Selección
        selected_population = tournament_selection(
            population,
            fitness_values,
            config.tournament_size
        )

        # Cruzamiento y mutación
        new_population = elites[:]

        index = 0

        while len(new_population) < config.population_size:
            parent1 = selected_population[index % len(selected_population)]
            parent2 = selected_population[(index + 1) % len(selected_population)]

            index += 2

            if random.random() <= config.crossover_probability:
                child1, child2 = ordered_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            if random.random() <= config.mutation_probability:
                child1 = swap_mutation(child1)

            if random.random() <= config.mutation_probability:
                child2 = swap_mutation(child2)

            new_population.append(child1)

            if len(new_population) < config.population_size:
                new_population.append(child2)

        population = new_population

    return {
        "seed": seed,
        "best_individual": best_individual,
        "best_fitness": best_fitness,
        "best_generation": best_generation,
        "log": log
    }


# ============================================================
# 8. Interpretación de la solución
# ============================================================

def explain_solution(individual):
    mapping = decode(individual)

    saturn = build_number(mapping, "SATURN")
    uranus = build_number(mapping, "URANUS")
    neptune = build_number(mapping, "NEPTUNE")
    pluto = build_number(mapping, "PLUTO")
    planets = build_number(mapping, "PLANETS")

    total = saturn + uranus + neptune + pluto

    is_valid = total == planets

    return mapping, saturn, uranus, neptune, pluto, planets, total, is_valid


# ============================================================
# 9. Guardado de logs
# ============================================================

def save_logs(results, filename="logs_saturn_uranus_neptune_pluto_planets.csv"):
    rows = []

    for result in results:
        for row in result["log"]:
            rows.append({
                "run_seed": row["run_seed"],
                "generation": row["generation"],
                "best_generation_fitness": row["best_generation_fitness"],
                "avg_generation_fitness": row["avg_generation_fitness"],
                "worst_generation_fitness": row["worst_generation_fitness"],
                "best_global_fitness": row["best_global_fitness"],
                "best_global_individual": row["best_global_individual"]
            })

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "run_seed",
                "generation",
                "best_generation_fitness",
                "avg_generation_fitness",
                "worst_generation_fitness",
                "best_global_fitness",
                "best_global_individual"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 10. Gráfico de comportamiento de aptitud
# ============================================================

def plot_best_run(best_result):
    try:
        import matplotlib.pyplot as plt

        generations = [row["generation"] for row in best_result["log"]]
        best_values = [row["best_generation_fitness"] for row in best_result["log"]]
        avg_values = [row["avg_generation_fitness"] for row in best_result["log"]]

        plt.figure(figsize=(10, 5))
        plt.plot(generations, best_values, label="Mejor fitness de la generación")
        plt.plot(generations, avg_values, label="Fitness promedio de la generación")
        plt.xlabel("Generación")
        plt.ylabel("Fitness")
        plt.title("Evolución de la función de aptitud - SATURN + URANUS + NEPTUNE + PLUTO = PLANETS")
        plt.legend()
        plt.grid(True)
        plt.show()

    except ImportError:
        print("matplotlib no está instalado. Se omite el gráfico.")


# ============================================================
# 11. Ejecución de varias corridas
# ============================================================

def main():
    config = Config()

    results = []

    for run in range(config.number_of_runs):
        result = run_genetic_algorithm(config, seed=run)
        results.append(result)

        mapping, saturn, uranus, neptune, pluto, planets, total, is_valid = explain_solution(
            result["best_individual"]
        )

        print("=" * 70)
        print(f"Corrida: {run}")
        print(f"Mejor fitness: {result['best_fitness']}")
        print(f"Generación del mejor individuo: {result['best_generation']}")
        print(f"Cromosoma: {result['best_individual']}")
        print(f"Asignación: {mapping}")
        print(f"SATURN  = {saturn}")
        print(f"URANUS  = {uranus}")
        print(f"NEPTUNE = {neptune}")
        print(f"PLUTO   = {pluto}")
        print(f"PLANETS = {planets}")
        print(f"Validación: {saturn} + {uranus} + {neptune} + {pluto} = {total}")
        print(f"¿Solución válida?: {is_valid}")

    best_result = min(results, key=lambda r: r["best_fitness"])

    print("\n" + "#" * 70)
    print("MEJOR RESULTADO GLOBAL")
    print("#" * 70)

    mapping, saturn, uranus, neptune, pluto, planets, total, is_valid = explain_solution(
        best_result["best_individual"]
    )

    print(f"Seed de la mejor corrida: {best_result['seed']}")
    print(f"Mejor fitness global: {best_result['best_fitness']}")
    print(f"Generación: {best_result['best_generation']}")
    print(f"Cromosoma: {best_result['best_individual']}")
    print(f"Asignación: {mapping}")
    print(f"SATURN  = {saturn}")
    print(f"URANUS  = {uranus}")
    print(f"NEPTUNE = {neptune}")
    print(f"PLUTO   = {pluto}")
    print(f"PLANETS = {planets}")
    print(f"Validación: {saturn} + {uranus} + {neptune} + {pluto} = {total}")
    print(f"¿Solución válida?: {is_valid}")

    save_logs(results)
    plot_best_run(best_result)


main()