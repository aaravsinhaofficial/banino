--[[ Open 11x11 square arena with the explore_goal_locations task semantics
(unmarked goal fixed within an episode, +10 on reach, agent respawns),
analogous to the square-arena experiments of Banino et al. 2018 (Fig. 2).
Generated as a single 11x11 room inside a 13x13 maze frame
(the maze generator requires odd sizes; the paper arena is 10x10 = 2.5 m), so the world
spans [0, 1300] units per axis (arena_cells = 13 for the env wrapper).
]]

local factory = require 'factories.explore.goal_locations_factory'

return factory.createLevelApi{
    mazeWidth = 13,
    mazeHeight = 13,
    roomCount = 1,
    roomMinSize = 11,
    roomMaxSize = 11,
    spawnCount = 10,
    objectCount = 4,
    decalFrequency = 0.2,
}
