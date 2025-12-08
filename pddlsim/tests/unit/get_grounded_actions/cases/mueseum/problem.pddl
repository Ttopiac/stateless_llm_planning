(define (problem bloomfield)
    (:domain mueseum)
    (:requirements :multiple-goals)
    (:objects lobby
              da-vinci-exhibit
              aerodynamics-exhibit
              automorphia-exhibit
              make-room-exhibit - exhibit
              emily
              bob - person)
    (:init (at bob lobby)
           (at emily da-vinci-exhibit))
    (:goals (at emily da-vinci-exhibit)
            (at emily aerodynamics-exhibit)
            (at bob make-room-exhibit)
            (at bob automorphia-exhibit)))