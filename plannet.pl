% Facts representing planets: planet(PlanetName, Mass, DistanceFromSun, OrbitalPeriod).
planet(mercury, 0.33, 57.9, 0.24).
planet(venus, 4.87, 108.2, 0.615).
planet(earth, 5.97, 149.6, 1.0).
planet(mars, 0.642, 227.9, 1.88).
planet(jupiter, 1898, 778.3, 11.86).
planet(saturn, 568, 1427, 29.46).
planet(uranus, 86.8, 2871, 84.01).
planet(neptune, 102, 4497.1, 164.8).
planet(pluto, 0.013, 5913, 248.0).
planet(planet_x, 50, 1200, 5.0).
planet(planet_y, 30, 300, 2.5).
planet(planet_z, 100, 800, 6.0).
planet(alpha_centauri_a, 20, 900, 4.0).
planet(alpha_centauri_b, 18, 700, 3.5).
planet(beta_tauri, 10, 650, 3.0).
planet(gamma_omicron, 25, 1200, 6.5).
planet(delta_pi, 15, 1000, 5.5).
planet(epsilon_alpha, 35, 1100, 4.5).
planet(zeta_omega, 55, 1350, 7.0).
planet(eta_tau, 65, 1400, 8.0).

% Query to find all planets
all_planets(List) :-
    findall(Planet, planet(Planet, _, _, _), List).

% Query to find a planet by its name
planet_by_name(Name, Mass, DistanceFromSun, OrbitalPeriod) :-
    planet(Name, Mass, DistanceFromSun, OrbitalPeriod).

% Query to find planets with mass greater than a given value
planets_with_mass_greater_than(Value, List) :-
    findall(Planet, (planet(Planet, Mass, _, _), Mass > Value), List).

% Query to find planets with orbital period greater than a given value
planets_with_orbital_period_greater_than(Value, List) :-
    findall(Planet, (planet(Planet, _, _, OrbitalPeriod), OrbitalPeriod > Value), List).

% Query to find planets in a given distance range
planets_in_distance_range(Min, Max, List) :-
    findall(Planet, (planet(Planet, _, Distance, _), Distance >= Min, Distance =< Max), List).

% Query to find planets ordered by distance from the sun
planets_ordered_by_distance(List) :-
    findall(Planet-Distance, planet(Planet, _, Distance, _), Pairs),
    sort(2, @=<, Pairs, SortedPairs),
    pairs_keys(SortedPairs, List).
