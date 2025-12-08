(define (problem darkest-dungeon)
    (:domain dungeon)
    (:requirements :revealables)
    (:objects start-room
              room-right
              room-top
              room-left
              goal-room - room
              dummy-switch
              switch-right
              goal-switch - switch
              bob - person)
    (:reveals (when (at bob room-right) (at switch-right room-right))
              (when (at bob room-top) (at goal-switch room-top)))
    (:init (at bob start-room)
           (on dummy-switch)
           (connected start-room room-right dummy-switch)
           (connected start-room room-top switch-right)
           (connected start-room room-left switch-right)
           (connected start-room goal-room goal-switch))
    (:goal (at bob goal-room)))