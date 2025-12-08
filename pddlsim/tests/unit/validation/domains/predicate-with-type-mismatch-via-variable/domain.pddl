(define (domain predicate-with-type-mismatch)
        (:requirements :typing)
        (:types cool beans)
        (:predicates (wow ?x - cool))
        (:action move
         :parameters (?b - beans)
         :precondition (wow ?b)))