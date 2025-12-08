(define (domain predicate-with-arity-mismatch)
        (:requirements :typing)
        (:types cool)
        (:predicates (wow ?x - cool))
        (:action move
         :parameters (?a ?b - cool)
         :precondition (wow ?a ?b)))