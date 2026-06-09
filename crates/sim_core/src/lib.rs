//! Pure battle simulation, built on the standalone `bevy_ecs`.
//!
//! Validates the §13 ECS design from the engine blueprint: units are entities,
//! the tick is a chained pipeline of systems, proximity goes through a spatial
//! index (not all-pairs), and the whole thing runs headless for tests.
//!
//! Fase 2c scope: terrain (passability, move cost, defensive bonus) + orders +
//! cooldowns, with movement now routed via **hexx A\*** around obstacles. Hex
//! math stays hand-rolled for the hot per-tick path; hexx is used where it
//! earns its place (pathfinding). The linked-list spatial index is still later.

use bevy_ecs::prelude::*;
use bevy_reflect::Reflect;
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Hex (flat-top axial). Hand-rolled math; hexx is used for A* (see movement).
// ---------------------------------------------------------------------------

#[derive(Component, Reflect, Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[reflect(Component)]
pub struct Hex {
    pub q: i32,
    pub r: i32,
}

const DIRS: [(i32, i32); 6] = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)];

impl Hex {
    pub fn new(q: i32, r: i32) -> Self {
        Self { q, r }
    }

    pub fn neighbors(self) -> [Hex; 6] {
        DIRS.map(|(dq, dr)| Hex::new(self.q + dq, self.r + dr))
    }

    /// Axial hex distance.
    pub fn distance(self, o: Hex) -> i32 {
        ((self.q - o.q).abs() + (self.q + self.r - o.q - o.r).abs() + (self.r - o.r).abs()) / 2
    }
}

// ---------------------------------------------------------------------------
// Terrain
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Terrain {
    Plains,
    Forest,
    Hill,
    Mountain,
    Water,
}

impl Terrain {
    pub fn passable(self) -> bool {
        !matches!(self, Terrain::Mountain | Terrain::Water)
    }
    /// Cooldown multiplier for entering this terrain.
    pub fn move_cost(self) -> u64 {
        match self {
            Terrain::Forest | Terrain::Hill => 2,
            _ => 1,
        }
    }
    /// Multiplier on damage *taken* while standing here (<1 = protective).
    pub fn defense_mult(self) -> f32 {
        match self {
            Terrain::Forest => 0.7,
            Terrain::Hill => 0.6,
            _ => 1.0,
        }
    }
}

#[derive(Resource, Default)]
pub struct TerrainMap {
    pub tiles: HashMap<(i32, i32), Terrain>,
}

impl TerrainMap {
    pub fn get(&self, h: Hex) -> Terrain {
        self.tiles.get(&(h.q, h.r)).copied().unwrap_or(Terrain::Plains)
    }
    pub fn set(&mut self, h: Hex, t: Terrain) {
        self.tiles.insert((h.q, h.r), t);
    }
}

/// Deterministic terrain over an axial rectangle. Coarse 4×4 blobs so biomes
/// clump instead of speckling. Pure hash noise — no dependencies, no RNG state.
pub fn generate_terrain(seed: i32, q_range: i32, r_range: i32) -> TerrainMap {
    let mut map = TerrainMap::default();
    for q in -q_range..=q_range {
        for r in -r_range..=r_range {
            let v = hash01(q.div_euclid(4), r.div_euclid(4), seed);
            let t = if v < 0.10 {
                Terrain::Water
            } else if v < 0.20 {
                Terrain::Mountain
            } else if v < 0.38 {
                Terrain::Forest
            } else if v < 0.52 {
                Terrain::Hill
            } else {
                Terrain::Plains
            };
            map.tiles.insert((q, r), t);
        }
    }
    map
}

fn hash01(a: i32, b: i32, seed: i32) -> f32 {
    let mut h = (a.wrapping_mul(73856093) ^ b.wrapping_mul(19349663) ^ seed.wrapping_mul(83492791))
        as u32;
    h ^= h >> 13;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 16;
    (h as f32) / (u32::MAX as f32)
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

#[derive(Component, Reflect, Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[reflect(Component)]
pub enum Team {
    Red,
    Blue,
}

#[derive(Component, Reflect, Clone, Copy, PartialEq, Eq, Debug)]
#[reflect(Component)]
pub enum Kind {
    Infantry,
    Cavalry,
    Skirmisher,
}

#[derive(Component, Reflect, Clone, Copy, Debug)]
#[reflect(Component)]
pub struct Health(pub f32);

/// Group this unit belongs to (1..=4), the unit of command for orders.
#[derive(Component, Reflect, Clone, Copy, Debug)]
#[reflect(Component)]
pub struct Group(pub u8);

/// Absolute tick this unit may next move on. Never reset on battle start
/// (the load-bearing invariant from hex-tactics).
#[derive(Component, Reflect, Clone, Copy, Default, Debug)]
#[reflect(Component)]
pub struct NextMove(pub u64);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Order {
    Idle,
    March,
    Charge,
    Hold,
}

pub const HOLD_REDUCTION: f32 = 0.5;
pub const VISION: i32 = 8;
/// A skirmisher within this many hexes of an enemy backs off instead of meleeing.
pub const KITE_THRESHOLD: i32 = 2;

/// Hexes a unit can strike: melee = 1, skirmishers shoot at range.
pub fn attack_range(kind: Kind) -> i32 {
    match kind {
        Kind::Skirmisher => 3,
        _ => 1,
    }
}

/// Base damage dealt per tick to a target in range.
pub fn attack_damage(kind: Kind) -> f32 {
    match kind {
        Kind::Infantry => 14.0,
        Kind::Cavalry => 11.0,
        Kind::Skirmisher => 8.0,
    }
}

/// Extra melee damage while charging (cavalry hits hardest; skirmishers never).
pub fn charge_bonus(kind: Kind, order: Order) -> f32 {
    if order != Order::Charge {
        return 0.0;
    }
    match kind {
        Kind::Cavalry => 16.0,
        Kind::Infantry => 8.0,
        Kind::Skirmisher => 0.0,
    }
}

pub fn max_hp(kind: Kind) -> f32 {
    match kind {
        Kind::Infantry => 100.0,
        Kind::Cavalry => 80.0,
        Kind::Skirmisher => 60.0,
    }
}

/// Ticks between moves. Cavalry/skirmishers are quicker; charging halves it.
fn move_period(kind: Kind, order: Order) -> u64 {
    let base = match kind {
        Kind::Infantry => 2,
        Kind::Cavalry => 1,
        Kind::Skirmisher => 1,
    };
    if order == Order::Charge {
        (base / 2).max(1)
    } else {
        base
    }
}

/// Component bundle for one unit. Shared by the headless tests and the Bevy
/// game (which adds render components alongside it on the same entity).
pub fn unit(team: Team, kind: Kind, hex: Hex, group: u8) -> impl Bundle {
    (hex, Health(max_hp(kind)), team, kind, Group(group), NextMove(0))
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

#[derive(Resource, Default)]
pub struct Tick(pub u64);

/// Per-(team, group) orders. Missing entries default to March.
#[derive(Resource, Default)]
pub struct Orders(pub HashMap<(Team, u8), Order>);

impl Orders {
    pub fn get(&self, team: Team, group: u8) -> Order {
        self.0.get(&(team, group)).copied().unwrap_or(Order::March)
    }
    pub fn set(&mut self, team: Team, group: u8, order: Order) {
        self.0.insert((team, group), order);
    }
}

/// One occupant per hex (rigid block). Rebuilt every tick. The cache-friendly
/// linked-list array version (§13) lands when we actually push to thousands.
#[derive(Resource, Default)]
pub struct SpatialIndex {
    pub cells: HashMap<(i32, i32), (Entity, Team)>,
}

impl SpatialIndex {
    fn at(&self, h: Hex) -> Option<(Entity, Team)> {
        self.cells.get(&(h.q, h.r)).copied()
    }
    fn occupied(&self, h: Hex) -> bool {
        self.cells.contains_key(&(h.q, h.r))
    }
}

/// Transient per-tick damage accumulator (scratch), keyed by target entity.
#[derive(Resource, Default)]
pub struct DamageBuffer(pub HashMap<Entity, f32>);

// ---------------------------------------------------------------------------
// Systems — pipeline order mirrors simulateTick's phases.
// ---------------------------------------------------------------------------

pub fn tick_and_clear(mut tick: ResMut<Tick>, mut dmg: ResMut<DamageBuffer>) {
    tick.0 += 1;
    dmg.0.clear();
}

pub fn build_spatial_index(units: Query<(Entity, &Hex, &Team)>, mut idx: ResMut<SpatialIndex>) {
    idx.cells.clear();
    for (e, h, t) in &units {
        idx.cells.insert((h.q, h.r), (e, *t));
    }
}

/// Melee units damage every adjacent enemy; skirmishers shoot the nearest enemy
/// within missile range. Charging melee attackers hit harder.
pub fn combat(
    units: Query<(&Hex, &Team, &Kind, &Group)>,
    orders: Res<Orders>,
    idx: Res<SpatialIndex>,
    mut dmg: ResMut<DamageBuffer>,
) {
    for (hex, team, kind, group) in &units {
        let order = orders.get(*team, group.0);
        let amount = attack_damage(*kind) + charge_bonus(*kind, order);
        if attack_range(*kind) == 1 {
            for n in hex.neighbors() {
                if let Some((enemy, eteam)) = idx.at(n) {
                    if eteam != *team {
                        *dmg.0.entry(enemy).or_insert(0.0) += amount;
                    }
                }
            }
        } else if let Some(enemy) = nearest_enemy_entity(&idx, *hex, *team, attack_range(*kind)) {
            *dmg.0.entry(enemy).or_insert(0.0) += amount;
        }
    }
}

/// Apply accumulated damage; defenders on protective terrain and Hold orders
/// take less. Despawn the dead before movement so movers skip corpses.
pub fn resolve_damage(
    mut commands: Commands,
    mut units: Query<(Entity, &mut Health, &Team, &Group, &Hex)>,
    orders: Res<Orders>,
    terrain: Res<TerrainMap>,
    dmg: Res<DamageBuffer>,
) {
    for (e, mut hp, team, group, hex) in &mut units {
        if let Some(d) = dmg.0.get(&e) {
            let mut incoming = *d * terrain.get(*hex).defense_mult();
            if orders.get(*team, group.0) == Order::Hold {
                incoming *= 1.0 - HOLD_REDUCTION;
            }
            hp.0 -= incoming;
        }
        if hp.0 <= 0.0 {
            commands.entity(e).despawn();
        }
    }
}

/// Disengaged units advance toward the enemy over passable terrain, gated by
/// cooldown (slower through forest/hills). A visible enemy is approached via
/// hexx A* (routes around mountains/water); otherwise a cheap greedy step
/// heads for the enemy line. A* is short-range only (target within VISION) to
/// stay cheap — at thousands of units it must be throttled/cached further.
pub fn movement(
    tick: Res<Tick>,
    orders: Res<Orders>,
    idx: Res<SpatialIndex>,
    terrain: Res<TerrainMap>,
    mut units: Query<(&mut Hex, &Team, &Kind, &Group, &mut NextMove)>,
) {
    let now = tick.0;
    let mut reserved: HashSet<(i32, i32)> = HashSet::new();

    for (mut hex, team, kind, group, mut next) in &mut units {
        let order = orders.get(*team, group.0);
        if matches!(order, Order::Hold | Order::Idle) {
            continue;
        }
        if now < next.0 {
            continue; // still on cooldown
        }

        let from = *hex;
        let enemy = nearest_enemy(&idx, from, *team);

        // Skirmishers kite: back off when an enemy gets close, shooting via combat.
        if *kind == Kind::Skirmisher {
            if let Some(e) = enemy {
                if from.distance(e) <= KITE_THRESHOLD {
                    if let Some(n) = kite_step(from, e, &terrain, &idx, &reserved) {
                        reserved.insert((n.q, n.r));
                        *hex = n;
                        next.0 = now + move_period(*kind, order) * terrain.get(n).move_cost();
                    }
                    continue;
                }
            }
        }

        // Melee: adjacent to an enemy → engaged, hold position (still fights).
        let fighting = from
            .neighbors()
            .iter()
            .any(|n| matches!(idx.at(*n), Some((_, t)) if t != *team));
        if fighting {
            continue;
        }

        let goal = enemy.unwrap_or_else(|| enemy_line(*team));

        // Prefer the A*-routed next hex toward a visible enemy; fall back to a
        // greedy step (toward the enemy line, or when the A* hex is taken).
        let step = match enemy.and_then(|e| a_star_step(from, e, &terrain)) {
            Some(s) if !idx.occupied(s) && !reserved.contains(&(s.q, s.r)) => Some(s),
            _ => greedy_step(from, goal, &terrain, &idx, &reserved),
        };

        if let Some(n) = step {
            reserved.insert((n.q, n.r));
            *hex = n;
            next.0 = now + move_period(*kind, order) * terrain.get(n).move_cost();
        }
    }
}

/// First hex after `from` on the A* path to `to`, routing around impassable
/// terrain. `None` if `to` is unreachable.
fn a_star_step(from: Hex, to: Hex, terrain: &TerrainMap) -> Option<Hex> {
    let start = hexx::Hex::new(from.q, from.r);
    let end = hexx::Hex::new(to.q, to.r);
    let path = hexx::algorithms::a_star(start, end, |_, b| {
        let t = terrain.get(Hex::new(b.x, b.y));
        if t.passable() {
            Some(t.move_cost() as u32)
        } else {
            None
        }
    })?;
    path.into_iter()
        .map(|h| Hex::new(h.x, h.y))
        .find(|h| *h != from)
}

/// Closest free, passable neighbor that reduces straight-line distance to the
/// goal. Used when there is no visible enemy, or the A* hex is occupied.
fn greedy_step(
    from: Hex,
    goal: Hex,
    terrain: &TerrainMap,
    idx: &SpatialIndex,
    reserved: &HashSet<(i32, i32)>,
) -> Option<Hex> {
    let mut best: Option<Hex> = None;
    let mut best_d = i32::MAX;
    for n in from.neighbors() {
        if !terrain.get(n).passable() || idx.occupied(n) || reserved.contains(&(n.q, n.r)) {
            continue;
        }
        let d = n.distance(goal);
        if d < best_d {
            best_d = d;
            best = Some(n);
        }
    }
    match best {
        Some(n) if n.distance(goal) < from.distance(goal) => Some(n),
        _ => None,
    }
}

/// Closest enemy within VISION, found by a range-bounded scan of the index
/// (not an all-pairs sweep) — the property that keeps the tick bounded at scale.
fn nearest_enemy(idx: &SpatialIndex, from: Hex, team: Team) -> Option<Hex> {
    let mut best = None;
    let mut best_d = i32::MAX;
    for dq in -VISION..=VISION {
        for dr in -VISION..=VISION {
            let h = Hex::new(from.q + dq, from.r + dr);
            let d = from.distance(h);
            if d == 0 || d > VISION {
                continue;
            }
            if let Some((_, t)) = idx.at(h) {
                if t != team && d < best_d {
                    best_d = d;
                    best = Some(h);
                }
            }
        }
    }
    best
}

/// Entity of the closest enemy within `range` (for ranged attacks).
fn nearest_enemy_entity(idx: &SpatialIndex, from: Hex, team: Team, range: i32) -> Option<Entity> {
    let mut best = None;
    let mut best_d = i32::MAX;
    for dq in -range..=range {
        for dr in -range..=range {
            let h = Hex::new(from.q + dq, from.r + dr);
            let d = from.distance(h);
            if d == 0 || d > range {
                continue;
            }
            if let Some((e, t)) = idx.at(h) {
                if t != team && d < best_d {
                    best_d = d;
                    best = Some(e);
                }
            }
        }
    }
    best
}

/// Free, passable neighbor that moves away from `enemy` (skirmisher kiting).
fn kite_step(
    from: Hex,
    enemy: Hex,
    terrain: &TerrainMap,
    idx: &SpatialIndex,
    reserved: &HashSet<(i32, i32)>,
) -> Option<Hex> {
    let mut best: Option<Hex> = None;
    let mut best_d = -1;
    for n in from.neighbors() {
        if !terrain.get(n).passable() || idx.occupied(n) || reserved.contains(&(n.q, n.r)) {
            continue;
        }
        let d = n.distance(enemy);
        if d > best_d {
            best_d = d;
            best = Some(n);
        }
    }
    match best {
        Some(n) if n.distance(enemy) >= from.distance(enemy) => Some(n),
        _ => None,
    }
}

/// Fallback march target: the far side of the field the enemy came from.
fn enemy_line(team: Team) -> Hex {
    match team {
        Team::Red => Hex::new(0, -30), // Red deploys at +r; enemy lies toward −r (up the screen)
        Team::Blue => Hex::new(0, 30), // Blue at −r; enemy toward +r (down)
    }
}

// ---------------------------------------------------------------------------
// Enemy AI (Blue). The game opts into this system; the headless `step` pipeline
// stays AI-free so tests drive orders directly.
// ---------------------------------------------------------------------------

/// Army-level order from the current force balance. Emulates the user's style:
/// advance to close, hold when even or losing, commit (charge) when ahead and
/// in contact.
pub fn ai_order(own: u32, foe: u32, engaged: u32) -> Order {
    if foe == 0 {
        Order::March // nothing to fight → advance
    } else if own * 5 < foe * 4 {
        Order::Hold // outnumbered (< 0.8×) → defend
    } else if engaged > 0 && own >= foe {
        Order::Charge // in contact and not behind → launch
    } else if engaged > 0 {
        Order::Hold // in contact, roughly even → hold the line
    } else {
        Order::March // not yet in contact → advance and amass
    }
}

/// Sets Blue's orders each tick from the battle state.
pub fn enemy_ai(units: Query<(&Hex, &Team)>, idx: Res<SpatialIndex>, mut orders: ResMut<Orders>) {
    let (mut own, mut foe, mut engaged) = (0u32, 0u32, 0u32);
    for (hex, team) in &units {
        match team {
            Team::Blue => {
                own += 1;
                if hex
                    .neighbors()
                    .iter()
                    .any(|n| matches!(idx.at(*n), Some((_, t)) if t == Team::Red))
                {
                    engaged += 1;
                }
            }
            Team::Red => foe += 1,
        }
    }
    orders.set(Team::Blue, 1, ai_order(own, foe, engaged));
}

// ---------------------------------------------------------------------------
// Headless driver — used by tests and re-used by the game's tick.
// ---------------------------------------------------------------------------

/// Run one full simulation tick on a bare `World`. No rendering, no app.
pub fn step(world: &mut World) {
    let mut schedule = Schedule::default();
    schedule.add_systems(
        (
            tick_and_clear,
            build_spatial_index,
            combat,
            resolve_damage,
            movement,
        )
            .chain(),
    );
    schedule.run(world);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_world() -> World {
        let mut w = World::new();
        w.insert_resource(Tick::default());
        w.insert_resource(Orders::default());
        w.insert_resource(TerrainMap::default());
        w.insert_resource(SpatialIndex::default());
        w.insert_resource(DamageBuffer::default());
        w
    }

    #[test]
    fn adjacent_enemies_deal_damage() {
        let mut w = fresh_world();
        let red = w.spawn(unit(Team::Red, Kind::Infantry, Hex::new(0, 0), 1)).id();
        w.spawn(unit(Team::Blue, Kind::Infantry, Hex::new(1, 0), 1));

        step(&mut w);

        let hp = w.get::<Health>(red).expect("red alive").0;
        assert!(hp < max_hp(Kind::Infantry), "red should have taken damage, hp={hp}");
    }

    #[test]
    fn isolated_unit_marches_toward_the_enemy_line() {
        let mut w = fresh_world();
        let blue = w.spawn(unit(Team::Blue, Kind::Cavalry, Hex::new(10, 0), 1)).id();

        step(&mut w);

        let h = *w.get::<Hex>(blue).expect("blue alive");
        assert!(h.q < 10, "blue should have advanced toward the enemy line, at {h:?}");
    }

    #[test]
    fn held_units_do_not_move() {
        let mut w = fresh_world();
        w.resource_mut::<Orders>().set(Team::Blue, 1, Order::Hold);
        let blue = w.spawn(unit(Team::Blue, Kind::Cavalry, Hex::new(10, 0), 1)).id();

        for _ in 0..10 {
            step(&mut w);
        }

        let h = *w.get::<Hex>(blue).expect("blue alive");
        assert_eq!(h, Hex::new(10, 0), "held unit must not move, at {h:?}");
    }

    #[test]
    fn hold_reduces_incoming_damage() {
        let dmg_taken = |order: Order| -> f32 {
            let mut w = fresh_world();
            w.resource_mut::<Orders>().set(Team::Red, 1, order);
            let red = w.spawn(unit(Team::Red, Kind::Infantry, Hex::new(0, 0), 1)).id();
            w.spawn(unit(Team::Blue, Kind::Infantry, Hex::new(1, 0), 1));
            step(&mut w);
            max_hp(Kind::Infantry) - w.get::<Health>(red).expect("red alive").0
        };
        assert!(
            dmg_taken(Order::Hold) < dmg_taken(Order::March),
            "Hold should reduce incoming damage"
        );
    }

    #[test]
    fn units_do_not_step_onto_impassable_terrain() {
        let mut w = fresh_world();
        // Blue at (10,0) wants to march toward -q. Wall the two left-ward
        // neighbors with mountains; it must not enter them.
        {
            let mut t = w.resource_mut::<TerrainMap>();
            t.set(Hex::new(9, 0), Terrain::Mountain);
            t.set(Hex::new(9, 1), Terrain::Mountain);
        }
        w.spawn(unit(Team::Blue, Kind::Cavalry, Hex::new(10, 0), 1));

        step(&mut w);

        let occupied: Vec<Hex> = {
            let mut q = w.query::<&Hex>();
            q.iter(&w).copied().collect()
        };
        assert!(
            !occupied.contains(&Hex::new(9, 0)) && !occupied.contains(&Hex::new(9, 1)),
            "unit must not step onto a mountain: {occupied:?}"
        );
    }

    #[test]
    fn a_star_routes_around_a_wall() {
        // Blue at (0,0); enemy at (3,0). A mountain wall at (1,0) and (1,1)
        // blocks the straight line, leaving a gap at (1,-1). The unit must step
        // around the wall, never onto a mountain.
        let mut w = fresh_world();
        {
            let mut t = w.resource_mut::<TerrainMap>();
            t.set(Hex::new(1, 0), Terrain::Mountain);
            t.set(Hex::new(1, 1), Terrain::Mountain);
        }
        let blue = w.spawn(unit(Team::Blue, Kind::Cavalry, Hex::new(0, 0), 1)).id();
        w.spawn(unit(Team::Red, Kind::Infantry, Hex::new(3, 0), 1));

        step(&mut w);

        let h = *w.get::<Hex>(blue).expect("blue alive");
        assert_ne!(h, Hex::new(0, 0), "unit should have moved");
        assert!(
            h != Hex::new(1, 0) && h != Hex::new(1, 1),
            "unit must route around the wall, not onto it: {h:?}"
        );
    }

    #[test]
    fn defensive_terrain_reduces_damage() {
        let dmg_on = |terrain: Terrain| -> f32 {
            let mut w = fresh_world();
            w.resource_mut::<TerrainMap>().set(Hex::new(0, 0), terrain);
            let red = w.spawn(unit(Team::Red, Kind::Infantry, Hex::new(0, 0), 1)).id();
            w.spawn(unit(Team::Blue, Kind::Infantry, Hex::new(1, 0), 1));
            step(&mut w);
            max_hp(Kind::Infantry) - w.get::<Health>(red).expect("red alive").0
        };
        assert!(
            dmg_on(Terrain::Hill) < dmg_on(Terrain::Plains),
            "defending on a hill should reduce damage taken"
        );
    }

    #[test]
    fn skirmishers_shoot_at_range_melee_does_not() {
        let dmg_to_enemy = |kind: Kind| -> f32 {
            let mut w = fresh_world();
            w.spawn(unit(Team::Red, kind, Hex::new(0, 0), 1));
            let blue = w.spawn(unit(Team::Blue, Kind::Infantry, Hex::new(3, 0), 1)).id();
            step(&mut w);
            max_hp(Kind::Infantry) - w.get::<Health>(blue).map(|h| h.0).unwrap_or(0.0)
        };
        assert!(dmg_to_enemy(Kind::Skirmisher) > 0.0, "skirmisher should hit at range 3");
        assert_eq!(dmg_to_enemy(Kind::Infantry), 0.0, "melee must not hit at range 3");
    }

    #[test]
    fn skirmishers_kite_from_adjacent_enemies() {
        let mut w = fresh_world();
        let sk = w.spawn(unit(Team::Red, Kind::Skirmisher, Hex::new(0, 0), 1)).id();
        w.spawn(unit(Team::Blue, Kind::Infantry, Hex::new(1, 0), 1));

        step(&mut w);

        let h = *w.get::<Hex>(sk).expect("skirmisher alive");
        assert!(h.distance(Hex::new(1, 0)) >= 2, "skirmisher should kite away, at {h:?}");
    }

    #[test]
    fn ai_advances_then_commits_and_defends() {
        assert_eq!(ai_order(10, 0, 0), Order::March, "no enemy → advance");
        assert_eq!(ai_order(10, 10, 0), Order::March, "even, no contact → advance");
        assert_eq!(ai_order(10, 10, 3), Order::Charge, "contact + not behind → launch");
        assert_eq!(ai_order(8, 10, 3), Order::Hold, "contact, slightly behind → hold");
        assert_eq!(ai_order(5, 10, 3), Order::Hold, "outnumbered → defend");
    }

    #[test]
    fn battle_resolves_to_a_decided_outcome() {
        let mut w = fresh_world();
        for r in 0..6 {
            w.spawn(unit(Team::Red, Kind::Infantry, Hex::new(0, r), 1));
            w.spawn(unit(Team::Blue, Kind::Infantry, Hex::new(1, r), 1));
        }

        for _ in 0..500 {
            step(&mut w);
        }

        let mut red = 0;
        let mut blue = 0;
        let mut q = w.query::<&Team>();
        for t in q.iter(&w) {
            match t {
                Team::Red => red += 1,
                Team::Blue => blue += 1,
            }
        }
        assert!(red == 0 || blue == 0, "one side should be wiped: red={red} blue={blue}");
    }
}
