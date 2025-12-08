(define (domain predicate-with-type-mismatch)
        (:predicates (a ?x))
        (:action move
         :parameters ()
         :precondition (a x)))