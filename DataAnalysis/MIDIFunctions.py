import pretty_midi
import numpy as np
import matplotlib.pyplot as plt




def get_velocities(onsets_times, pm):

    velocities = []
    for onset in onsets_times:

        closest_note = None
        closest_distance = float('inf')
        
        for note in pm.instruments[0].notes:  # Considera solo il primo strumento (batteria monofonica)
            note_distance = abs(note.start - onset)
            if note_distance < closest_distance:
                closest_note = note
                closest_distance = note_distance
        
        if closest_note is not None:
            velocity = closest_note.velocity
            # print(f"Velocità della nota più vicina all'onset {onset}: {velocity}")
        else:
            print(f"Nessuna nota trovata per l'onset {onset}")

        velocities.append(velocity)

    onsets_velocities = np.array(velocities)
    onsets_velocities = onsets_velocities/127
    # print("The onsets have velocities: ", onsets_velocities)

    return onsets_velocities