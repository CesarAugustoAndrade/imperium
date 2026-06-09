//! Imperium — Fase 1 spike.
//!
//! A Bevy 0.18 window that runs the pure `sim_core` battle on a fixed 2 Hz tick
//! and renders each unit as a colored hexagon. Two infantry blocks advance,
//! clash, and one side is wiped. Press 1/2/3 to order the RED army to
//! March / Charge / Hold. All logic lives in `sim_core`; the renderer just
//! mirrors `Hex` → `Transform` each frame.

use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::remote::{http::RemoteHttpPlugin, RemotePlugin};
use sim_core::{
    generate_terrain, unit, DamageBuffer, Group, Health, Hex, Kind, NextMove, Order, Orders,
    SpatialIndex, Team, Terrain, TerrainMap, Tick,
};

const HEX_SIZE: f32 = 12.0;
const ARMY_COLS: i32 = 28;
const ARMY_ROWS: i32 = 54;
const GAP: i32 = 12; // hexes between the two armies' inner edges
const GRID_Q: i32 = 40;
const GRID_R: i32 = 34;
const SEED: i32 = 7;
/// Rotate a Bevy `RegularPolygon` (pointy-top by default) to flat-top.
const FLAT_TOP: f32 = std::f32::consts::FRAC_PI_6;
const TERRAIN_Z: f32 = -10.0;
/// Tiny per-screen-row z step: front rows (lower on screen) sort above the
/// cliff walls of the rows behind them — the 2.5D "viewed from below" stacking.
const Z_BY_ROW: f32 = 0.001;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "IMPERIUM".into(),
                ..default()
            }),
            ..default()
        }))
        // Bevy Remote Protocol: query/mutate the live ECS over JSON-RPC (port
        // 15702). This is the runtime-agent hook — an agent can drive/inspect
        // the running battle. Register the sim components so they're queryable.
        .add_plugins((RemotePlugin::default(), RemoteHttpPlugin::default()))
        // FPS + frame time logged to console every second (scale stress test).
        .add_plugins((
            FrameTimeDiagnosticsPlugin::default(),
            LogDiagnosticsPlugin::default(),
        ))
        .register_type::<Hex>()
        .register_type::<Health>()
        .register_type::<Team>()
        .register_type::<Kind>()
        .register_type::<Group>()
        .register_type::<NextMove>()
        .insert_resource(ClearColor(Color::srgb(0.04, 0.05, 0.07)))
        .insert_resource(generate_terrain(SEED, GRID_Q, GRID_R))
        // Battle sim runs on a fixed timestep, decoupled from render framerate.
        .insert_resource(Time::<Fixed>::from_hz(2.0))
        .insert_resource(Tick::default())
        .insert_resource(Orders::default())
        .insert_resource(SpatialIndex::default())
        .insert_resource(DamageBuffer::default())
        .add_systems(Startup, setup)
        .add_systems(
            FixedUpdate,
            (
                sim_core::tick_and_clear,
                sim_core::build_spatial_index,
                sim_core::enemy_ai,
                sim_core::combat,
                sim_core::resolve_damage,
                sim_core::movement,
                log_status,
            )
                .chain(),
        )
        .add_systems(Update, (control, sync_transforms))
        .run();
}

fn terrain_color(t: Terrain) -> Color {
    match t {
        Terrain::Plains => Color::srgb(0.20, 0.28, 0.17),
        Terrain::Forest => Color::srgb(0.11, 0.21, 0.12),
        Terrain::Hill => Color::srgb(0.36, 0.29, 0.18),
        Terrain::Mountain => Color::srgb(0.36, 0.36, 0.40),
        Terrain::Water => Color::srgb(0.11, 0.21, 0.42),
    }
}

/// Elevation in pixels — drives both the upward offset of the hex top and the
/// height of its cliff wall. Mirrors hex-tactics: water flat, mountains tower.
fn terrain_height(t: Terrain) -> f32 {
    match t {
        Terrain::Water => 0.0,
        Terrain::Plains => 3.0,
        Terrain::Forest => 12.0,
        Terrain::Hill => 32.0,
        Terrain::Mountain => 60.0,
    }
}

/// Darkened terrain color for the shaded cliff face.
fn wall_color(t: Terrain) -> Color {
    let c = terrain_color(t).to_srgba();
    Color::srgb(c.red * 0.45, c.green * 0.45, c.blue * 0.45)
}

/// Stable per-tile hash → picks a shade variant (fakes texture).
fn tile_hash(q: i32, r: i32) -> u32 {
    let mut h = (q.wrapping_mul(374761393) ^ r.wrapping_mul(668265263)) as u32;
    h ^= h >> 13;
    h = h.wrapping_mul(1274126177);
    h ^ (h >> 16)
}

/// Flat-top axial → world pixels.
fn hex_to_world(h: Hex) -> Vec2 {
    let x = HEX_SIZE * 1.5 * h.q as f32;
    let y = HEX_SIZE * 3.0_f32.sqrt() * (h.r as f32 + h.q as f32 / 2.0);
    Vec2::new(x, -y)
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut terrain: ResMut<TerrainMap>,
) {
    commands.spawn(Camera2d);

    // Carve the two deploy zones to Plains so units never spawn stuck.
    for col in 0..ARMY_COLS {
        for row in 0..ARMY_ROWS {
            let r = row - ARMY_ROWS / 2;
            terrain.set(Hex::new(-(GAP / 2) - 1 - col, r), Terrain::Plains);
            terrain.set(Hex::new((GAP / 2) + 1 + col, r), Terrain::Plains);
        }
    }

    // A winding river down the central no-man's-land, with fords (land bridges)
    // every few rows so the armies can still cross and clash. Water is flat →
    // reads as a channel cut into the raised land.
    for r in -GRID_R..=GRID_R {
        if r.rem_euclid(10) < 2 {
            continue; // ford
        }
        let cq = (6.0 * (r as f32 * 0.22).sin()) as i32;
        for dq in 0..=1 {
            let h = Hex::new(cq + dq, r);
            if h.distance(Hex::new(0, 0)) > 1 {
                terrain.set(h, Terrain::Water); // keep the central objective on land
            }
        }
    }

    // 2.5D terrain: each hex is a flat top raised by its elevation, with a dark
    // south-facing cliff wall below. Tiles z-sort by screen row so front tiles
    // overlap the walls of those behind — the "viewed from below" look.
    let top_mesh = meshes.add(RegularPolygon::new(HEX_SIZE * 0.96, 6));
    let wall_mesh = meshes.add(Rectangle::new(1.0, 1.0));
    let terrains = [
        Terrain::Plains,
        Terrain::Forest,
        Terrain::Hill,
        Terrain::Mountain,
        Terrain::Water,
    ];
    // Per-terrain shade variants → per-tile tonal jitter that fakes texture.
    let top_variants: Vec<(Terrain, Vec<Handle<ColorMaterial>>)> = terrains
        .iter()
        .map(|&t| {
            let c = terrain_color(t).to_srgba();
            let v = [0.80_f32, 0.90, 1.0, 1.10, 1.20]
                .iter()
                .map(|&f| {
                    materials.add(Color::srgb(
                        (c.red * f).min(1.0),
                        (c.green * f).min(1.0),
                        (c.blue * f).min(1.0),
                    ))
                })
                .collect();
            (t, v)
        })
        .collect();
    let wall_mats: Vec<(Terrain, Handle<ColorMaterial>)> = terrains
        .iter()
        .map(|&t| (t, materials.add(wall_color(t))))
        .collect();
    let top_for = |t: Terrain, q: i32, r: i32| {
        let v = &top_variants.iter().find(|(k, _)| *k == t).unwrap().1;
        v[tile_hash(q, r) as usize % v.len()].clone()
    };
    let wall_for = |t: Terrain| wall_mats.iter().find(|(k, _)| *k == t).unwrap().1.clone();

    let hex_half_h = HEX_SIZE * 3.0_f32.sqrt() / 2.0;
    let wall_w = HEX_SIZE * 1.5;
    for q in -GRID_Q..=GRID_Q {
        for r in -GRID_R..=GRID_R {
            let t = terrain.get(Hex::new(q, r));
            let base = hex_to_world(Hex::new(q, r));
            let h = terrain_height(t);
            let z = TERRAIN_Z + (-base.y) * Z_BY_ROW;
            if h > 0.5 {
                let wall_h = hex_half_h + h;
                let wy = (base.y + h) - wall_h / 2.0;
                commands.spawn((
                    Mesh2d(wall_mesh.clone()),
                    MeshMaterial2d(wall_for(t)),
                    Transform::from_xyz(base.x, wy, z).with_scale(Vec3::new(wall_w, wall_h, 1.0)),
                ));
            }
            commands.spawn((
                Mesh2d(top_mesh.clone()),
                MeshMaterial2d(top_for(t, q, r)),
                Transform::from_xyz(base.x, base.y + h, z + 0.0005)
                    .with_rotation(Quat::from_rotation_z(FLAT_TOP)),
            ));
        }
    }

    // Central objective ("mid"): a tall gold marker at the map centre, around
    // which the battle is fought.
    {
        let h = 42.0;
        let base = hex_to_world(Hex::new(0, 0));
        let z = TERRAIN_Z + (-base.y) * Z_BY_ROW + 0.05;
        let wall_h = hex_half_h + h;
        commands.spawn((
            Mesh2d(wall_mesh.clone()),
            MeshMaterial2d(materials.add(Color::srgb(0.45, 0.38, 0.12))),
            Transform::from_xyz(base.x, (base.y + h) - wall_h / 2.0, z)
                .with_scale(Vec3::new(wall_w, wall_h, 1.0)),
        ));
        commands.spawn((
            Mesh2d(meshes.add(RegularPolygon::new(HEX_SIZE, 6))),
            MeshMaterial2d(materials.add(Color::srgb(0.88, 0.74, 0.28))),
            Transform::from_xyz(base.x, base.y + h, z + 0.001)
                .with_rotation(Quat::from_rotation_z(FLAT_TOP)),
        ));
    }

    let mesh = meshes.add(RegularPolygon::new(HEX_SIZE * 0.42, 6));
    // One material per (team, kind): team hue, brightness by kind.
    let mut umat: Vec<(Team, Kind, Handle<ColorMaterial>)> = Vec::new();
    for team in [Team::Red, Team::Blue] {
        for kind in [Kind::Infantry, Kind::Cavalry, Kind::Skirmisher] {
            umat.push((team, kind, materials.add(unit_color(team, kind))));
        }
    }
    let mat_for = |team, kind| umat.iter().find(|(t, k, _)| *t == team && *k == kind).unwrap().2.clone();

    // Red on the left, Blue on the right; a gap in the middle. Cavalry forms
    // the front (inner columns), infantry the centre, skirmishers the rear.
    let mut n = 0;
    for col in 0..ARMY_COLS {
        let kind = kind_for(col, ARMY_COLS);
        for row in 0..ARMY_ROWS {
            let r = row - ARMY_ROWS / 2;
            let (rq, bq) = (-(GAP / 2) - 1 - col, (GAP / 2) + 1 + col);
            spawn_unit(&mut commands, &mesh, &mat_for(Team::Red, kind), Team::Red, kind, Hex::new(rq, r));
            spawn_unit(&mut commands, &mesh, &mat_for(Team::Blue, kind), Team::Blue, kind, Hex::new(bq, r));
            n += 2;
        }
    }

    info!("spawned {n} units | controls: [1] Red March  [2] Red Charge  [3] Red Hold");
}

fn spawn_unit(
    commands: &mut Commands,
    mesh: &Handle<Mesh>,
    material: &Handle<ColorMaterial>,
    team: Team,
    kind: Kind,
    hex: Hex,
) {
    let p = hex_to_world(hex);
    commands.spawn((
        unit(team, kind, hex, 1),
        Mesh2d(mesh.clone()),
        MeshMaterial2d(material.clone()),
        Transform::from_xyz(p.x, p.y, 0.0).with_rotation(Quat::from_rotation_z(FLAT_TOP)),
    ));
}

/// Front line is cavalry, centre infantry, rear skirmishers — by how deep the
/// column sits in the formation (col 0 = inner/front).
fn kind_for(col: i32, cols: i32) -> Kind {
    if col < cols / 4 {
        Kind::Cavalry
    } else if col >= cols * 3 / 4 {
        Kind::Skirmisher
    } else {
        Kind::Infantry
    }
}

fn unit_color(team: Team, kind: Kind) -> Color {
    let (r, g, b): (f32, f32, f32) = match team {
        Team::Red => (0.92, 0.30, 0.30),
        Team::Blue => (0.34, 0.52, 0.96),
    };
    let f: f32 = match kind {
        Kind::Cavalry => 1.25,
        Kind::Infantry => 1.0,
        Kind::Skirmisher => 0.65,
    };
    Color::srgb((r * f).min(1.0), (g * f).min(1.0), (b * f).min(1.0))
}

/// Keyboard → orders for the Red army (group 1).
fn control(keys: Res<ButtonInput<KeyCode>>, mut orders: ResMut<Orders>) {
    if keys.just_pressed(KeyCode::Digit1) {
        orders.set(Team::Red, 1, Order::March);
        info!("Red → March");
    }
    if keys.just_pressed(KeyCode::Digit2) {
        orders.set(Team::Red, 1, Order::Charge);
        info!("Red → Charge");
    }
    if keys.just_pressed(KeyCode::Digit3) {
        orders.set(Team::Red, 1, Order::Hold);
        info!("Red → Hold");
    }
}

/// Mirror the sim's authoritative `Hex` onto the render `Transform` each frame,
/// lifting each unit onto the elevation of the tile it stands on and z-sorting
/// it above the terrain (front units over back units).
fn sync_transforms(mut q: Query<(&Hex, &mut Transform)>, terrain: Res<TerrainMap>) {
    for (h, mut t) in &mut q {
        let p = hex_to_world(*h);
        let elev = terrain_height(terrain.get(*h));
        t.translation.x = p.x;
        t.translation.y = p.y + elev + 3.0;
        t.translation.z = 1.0 + (-p.y) * Z_BY_ROW;
    }
}

fn log_status(tick: Res<Tick>, orders: Res<Orders>, q: Query<&Team>) {
    if tick.0 % 4 != 0 {
        return;
    }
    let (mut red, mut blue) = (0, 0);
    for t in &q {
        match t {
            Team::Red => red += 1,
            Team::Blue => blue += 1,
        }
    }
    info!(
        "tick {:>4} | red {:>3} ({:?}) | blue {:>3} (AI {:?})",
        tick.0,
        red,
        orders.get(Team::Red, 1),
        blue,
        orders.get(Team::Blue, 1),
    );
}
