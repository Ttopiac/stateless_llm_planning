(define (problem darkest-dungeon)
    (:domain dungeon)
    (:objects start-room
              room-right
              room-top
              room-left
              goal-room - room
              dummy-switch
              switch-right
              goal-switch - switch
              bob - person)
    (:init (at bob start-room)
           (on dummy-switch)
           (connected start-room room-right dummy-switch)
           (at switch-right room-right)
           (connected start-room room-top switch-right)
           (connected start-room room-left switch-right)
           (at goal-switch room-top)
           (connected start-room goal-room goal-switch))
    (:goal (at bob goal-room)))