#pragma once

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <Functions/keyvaluepair/src/impl/state/KeyStateHandler.h>
#include <Functions/keyvaluepair/src/impl/state/ValueStateHandler.h>
#include <Functions/keyvaluepair/src/impl/KeyValuePairEscapingProcessor.h>
#include <Functions/keyvaluepair/src/KeyValuePairExtractor.h>

namespace DB
{

/*
 * Implements key value pair extraction by ignoring escaping and deferring its processing to the end.
 * This strategy allows more efficient memory usage in case of very noisy files because it does not have to
 * store characters while reading an element. Because of that, std::string_views can be used to store key value pairs.
 *
 * In the end, the unescaped key value pair views are converted into escaped key value pairs. At this stage, memory is allocated
 * to store characters, but noise is no longer an issue.
 * */
class LazyEscapingKeyValuePairExtractor : public KeyValuePairExtractor {
public:
    LazyEscapingKeyValuePairExtractor(KeyStateHandler keyStateHandler, ValueStateHandler valueStateHandler, KeyValuePairEscapingProcessor keyValuePairEscapingProcessor);

    [[nodiscard]] Response extract(const std::string & file) override;

private:
    NextState extract(const std::string & file, std::size_t pos, State state);

    NextState flushPair(const std::string & file, std::size_t pos);

    KeyStateHandler key_state_handler;
    ValueStateHandler value_state_handler;
    KeyValuePairEscapingProcessor escaping_processor;

    std::unordered_map<std::string_view, std::string_view> response_views;
};

}

